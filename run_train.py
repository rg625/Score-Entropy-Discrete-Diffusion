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
from eval_external import ExternalPerplexityCallback


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
    custom_tokenizer_path = os.path.join("datasets", cfg.data.train, "tokenizer_bpe_16k.json")
    
    if os.path.exists(custom_tokenizer_path):
        mprint(f"Loading custom tokenizer from: {custom_tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tokenizer_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = "[PAD]"
        if tokenizer.eos_token is None: tokenizer.eos_token = "[EOS]"
        
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
    
    # ------------------------------------------------------------------
    # MODEL & GRAPH
    # ------------------------------------------------------------------
    graph = graph_lib.get_graph(cfg, device)
    
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)

    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5

    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    scaler = torch.cuda.amp.GradScaler()
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 

    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)

    if cfg.training.snapshot_sampling:
        # Note: ensures we request indices for External PPL
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    # ------------------------------------------------------------------
    # NEW: EXTERNAL EVALUATOR SETUP
    # ------------------------------------------------------------------
    ar_ckpt_path = "./checkpoints_ar/ar_best.pt"
    external_evaluator = None
    
    # Only Rank 0 needs to hold the AR model to save VRAM on others
    if rank == 0:
        if os.path.exists(ar_ckpt_path):
            mprint(f"initializing External Evaluator using {ar_ckpt_path}")
            external_evaluator = ExternalPerplexityCallback(
                ar_checkpoint_path=ar_ckpt_path,
                vocab_size=cfg.tokens, 
                sequence_len=cfg.model.length,
                device=device
            )
        else:
            mprint("WARNING: AR Oracle Checkpoint not found. External PPL will be skipped.")

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    while state['step'] < num_train_steps + 1:
        step = state['step']

        # --- Train Step ---
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
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

            # --- Internal Eval Step ---
            if step % cfg.training.eval_freq == 0:
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

            # --- Snapshot & External Evaluation ---
            if (step > 0 and step % cfg.training.snapshot_freq == 0) or step == num_train_steps:
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")
                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    # Swap to EMA weights
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    
                    # 1. Generate Raw Samples (Indices)
                    sample_indices = sampling_fn(score_model)
                    
                    # 2. Decode for inspection
                    sentences = tokenizer.batch_decode(sample_indices)
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n" + "="*50 + "\n")

                    if rank == 0 and writer is not None:
                        md_text = "\n\n---\n\n".join([f"**Sample {i+1}:**\n{s}" for i, s in enumerate(sentences[:4])])
                        writer.add_text("samples", md_text, step)

                    # 3. EXTERNAL PERPLEXITY EVALUATION (Rank 0 only)
                    if rank == 0 and external_evaluator is not None:
                        mprint("Starting External PPL Eval...")
                        
                        # Wrapper to allow the callback to call sampling_fn independently
                        # We use a lambda to inject the current model state
                        def specific_sampling_fn(m):
                            with torch.no_grad():
                                return sampling_fn(m)

                        # Note: We pass eval_ds directly (dataset object) or an iterator
                        # We create a fresh loader for the evaluator to ensure we get valid samples
                        # without messing up the main training iterator
                        val_loader_fresh = torch.utils.data.DataLoader(
                            eval_ds.dataset, batch_size=16, shuffle=False
                        )

                        gen_ppl, real_ppl = external_evaluator.evaluate(
                            diffusion_model=score_model,
                            sampling_fn=specific_sampling_fn,
                            valid_loader=val_loader_fresh,
                            num_samples=1024  # As requested
                        )
                        
                        if writer is not None:
                            writer.add_scalar("val/external_ppl_gen", gen_ppl, step)
                            writer.add_scalar("val/external_ppl_real", real_ppl, step)
                            writer.add_scalar("val/external_ppl_gap", gen_ppl - real_ppl, step)

                    # Restore Training Weights
                    ema.restore(score_model.parameters())

                    dist.barrier()
    
    if rank == 0 and writer is not None:
        writer.close()