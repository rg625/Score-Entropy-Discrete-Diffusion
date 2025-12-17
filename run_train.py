import datetime
import os
import os.path
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, PreTrainedTokenizerFast


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # --- TENSORBOARD SETUP (Rank 0 Only) ---
    writer = None
    if rank == 0:
        tb_log_dir = os.path.join(work_dir, "tensorboard")
        utils.makedirs(tb_log_dir)
        writer = SummaryWriter(log_dir=tb_log_dir)

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    
    # ------------------------------------------------------------------
    # TOKENIZER SETUP 
    # ------------------------------------------------------------------
    # Check if we have a custom tokenizer trained for this dataset
    custom_tokenizer_path = os.path.join("datasets", cfg.data.train, "tokenizer_bpe_16k.json")
    
    if os.path.exists(custom_tokenizer_path):
        mprint(f"Loading custom tokenizer from: {custom_tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tokenizer_path)
        # Ensure special tokens exist
        if tokenizer.pad_token is None: tokenizer.pad_token = "[PAD]"
        if tokenizer.eos_token is None: tokenizer.eos_token = "[EOS]"
        
        # Update config to match the actual vocab size
        old_tokens = cfg.tokens
        cfg.tokens = len(tokenizer)
        mprint(f"Updated cfg.tokens from {old_tokens} -> {cfg.tokens} to match custom tokenizer.")
    else:
        mprint("No custom tokenizer found. Falling back to default GPT2TokenizerFast.")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']

        # --- Train Step ---
        batch_data = next(train_iter)
        batch = batch_data['input_ids'].to(device)
        
        loss = train_step_fn(state, batch)

        # Log on new step
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                
                if rank == 0 and writer is not None:
                    writer.add_scalar("train/loss", loss.item(), step)
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                # --- Eval Step ---
                try:
                    eval_batch_data = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_ds)
                    eval_batch_data = next(eval_iter)
                
                eval_batch = eval_batch_data['input_ids'].to(device)
                
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

                if rank == 0 and writer is not None:
                    writer.add_scalar("val/loss", eval_loss.item(), step)

            # --- BPC Evaluation (Every 100k steps) ---
            if step > 0 and step % 100000 == 0:
                mprint(f"Starting BPC evaluation at step {step}...")
                n_bpc_batches = 50 
                total_nats = 0.0
                total_chars = 0
                
                with torch.no_grad():
                    for i in range(n_bpc_batches):
                        try:
                            bpc_batch_data = next(eval_iter)
                        except StopIteration:
                            eval_iter = iter(eval_ds)
                            bpc_batch_data = next(eval_iter)
                        
                        bpc_batch = bpc_batch_data['input_ids'].to(device)
                        batch_size = bpc_batch.shape[0]

                        # eval_step_fn returns mean Nats/Seq
                        avg_nats_per_seq = eval_step_fn(state, bpc_batch)
                        local_batch_nats = avg_nats_per_seq * batch_size

                        # Count characters
                        cpu_tokens = bpc_batch.cpu().numpy()
                        decoded_texts = tokenizer.batch_decode(cpu_tokens)
                        local_char_count = sum(len(text) for text in decoded_texts)

                        # Aggregate
                        metrics_tensor = torch.tensor([local_batch_nats.item(), local_char_count], device=device)
                        dist.all_reduce(metrics_tensor)
                        
                        total_nats += metrics_tensor[0].item()
                        total_chars += metrics_tensor[1].item()

                total_bits = total_nats / np.log(2)
                bpc = total_bits / total_chars
                
                mprint(f"Step: {step} | Test BPC (over {n_bpc_batches} batches): {bpc:.4f}")
                
                if rank == 0 and writer is not None:
                    writer.add_scalar("val/bpc", bpc, step)
                
                dist.barrier()

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if rank == 0 and writer is not None:
                        md_text = "\n\n---\n\n".join([f"**Sample {i+1}:**\n{s}" for i, s in enumerate(sentences)])
                        writer.add_text("samples", md_text, step)

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            
                            n_samples = sample.shape[0]
                            ppl_bs = cfg.eval.perplexity_batch_size
                            total_perplexity = 0.0
                            num_batches_processed = 0
                            
                            for i in range(0, n_samples, ppl_bs):
                                s = sample[i : i + ppl_bs]
                                if s.shape[0] == 0: continue

                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                                num_batches_processed += 1
                            
                            if num_batches_processed > 0:
                                total_perplexity /= num_batches_processed
                            
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                            if rank == 0 and writer is not None:
                                writer.add_scalar("val/perplexity", total_perplexity, step)

                            del eval_model, logits, loss

                    dist.barrier()
    
    if rank == 0 and writer is not None:
        writer.close()