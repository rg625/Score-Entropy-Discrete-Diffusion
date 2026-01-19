import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
import os
import numpy as np

class ExternalPerplexityCallback:
    def __init__(self, ar_checkpoint_path, vocab_size, sequence_len, device):
        """
        Evaluator using a pretrained Auto-Regressive model.
        
        Args:
            ar_checkpoint_path: Path to the .pt file of the trained AR model.
            vocab_size: Must match the diffusion model (65536).
            sequence_len: Context window (e.g. 512 or 1024).
            device: torch.device
        """
        self.device = device
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        
        if not os.path.exists(ar_checkpoint_path):
            print(f"⚠️ WARNING: AR Checkpoint not found at {ar_checkpoint_path}. External PPL will fail.")
            self.ar_model = None
        else:
            print(f"⚖️ Evaluator: Loading AR oracle from {ar_checkpoint_path}...")
            # Must match the config used in train_ar.py
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=sequence_len,
                n_ctx=sequence_len,
                n_embd=768, n_layer=12, n_head=12,
                use_cache=False
            )
            self.ar_model = GPT2LMHeadModel(config).to(device)
            
            # Load weights safely
            state_dict = torch.load(ar_checkpoint_path, map_location=device)
            # Handle cases where state_dict might have "module." prefix if saved from DDP
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.ar_model.load_state_dict(new_state_dict)
            self.ar_model.eval()
            print("✅ Evaluator: AR Model loaded successfully.")

    def compute_perplexity(self, samples):
        """
        Computes PPL of samples using the AR model.
        samples: [B, SeqLen] (LongTensor of token IDs)
        """
        if self.ar_model is None: return 0.0, 0.0

        # Ensure type and range
        samples = samples.long().to(self.device)
        
        # Clamp just in case, though diffusion shouldn't output OOB
        samples = torch.clamp(samples, 0, self.vocab_size - 1)
        
        with torch.no_grad():
            # GPT2 forward with labels returns CrossEntropy loss (base e)
            outputs = self.ar_model(samples, labels=samples)
            nll = outputs.loss 
            ppl = torch.exp(nll)
            
        return ppl.item(), nll.item()

    @torch.no_grad()
    def evaluate(self, diffusion_model, sampling_fn, valid_loader, num_samples=1024, batch_size=16):
        """
        Generates samples from diffusion model, calculates their AR-PPL.
        Also calculates AR-PPL on real validation data (Baseline).
        
        Args:
            diffusion_model: The model being trained.
            sampling_fn: Function taking (model) -> [Batch, Seq] tokens.
            valid_loader: DataLoader for real data baseline.
            num_samples: Total samples to evaluate.
        """
        if self.ar_model is None: return 0.0, 0.0

        print(f"\n--- 🧪 External Perplexity Evaluation ({num_samples} samples) ---")
        
        # 1. Generate Samples from Discrete Diffusion
        diffusion_model.eval()
        generated_batches = []
        samples_collected = 0
        
        print("   > Generating samples from Diffusion Model...")
        while samples_collected < num_samples:
            # Run the sampling function provided by the training loop
            # This function encapsulates the diffusion reverse process
            batch_sample = sampling_fn(diffusion_model) 
            
            # Ensure it's on the right device/dtype
            batch_sample = batch_sample.to(self.device)
            generated_batches.append(batch_sample)
            
            samples_collected += batch_sample.shape[0]
            if samples_collected >= num_samples:
                break
                
        all_gen = torch.cat(generated_batches, dim=0)[:num_samples]
        
        # 2. Evaluate Generated Samples
        print("   > Computing PPL on Generated Samples...")
        gen_ppl_accum = 0.0
        n_batches = 0
        
        # Process in mini-batches to save VRAM on the AR model side
        for i in range(0, len(all_gen), batch_size):
            batch = all_gen[i:i+batch_size]
            ppl, _ = self.compute_perplexity(batch)
            gen_ppl_accum += ppl
            n_batches += 1
            
        avg_gen_ppl = gen_ppl_accum / n_batches
        
        # 3. Evaluate Real Data (Baseline)
        print("   > Computing PPL on Real Validation Data (Baseline)...")
        real_ppl_accum = 0.0
        n_real_batches = 0
        real_samples_seen = 0
        
        for batch_data in valid_loader:
            # Handle dict/tensor variance
            if isinstance(batch_data, dict):
                batch = batch_data.get('input_ids', batch_data.get('text', None))
            else:
                batch = batch_data
            
            if batch is None: continue
            
            batch = batch.to(self.device)
            
            # If batch is token bits (binary), we can't use it for AR eval easily
            # But valid_loader for Diffusion likely yields tokens now if we unified configs.
            # If it yields bits, we skip real data baseline or need a decoder.
            if batch.dtype == torch.float:
                # We are in binary mode. Skip Real Baseline for now unless decoding is available.
                # Assuming user handles dataset config correctly.
                pass
            else:
                ppl, _ = self.compute_perplexity(batch)
                real_ppl_accum += ppl
                n_real_batches += 1
                real_samples_seen += batch.shape[0]
            
            if real_samples_seen >= num_samples:
                break
        
        avg_real_ppl = real_ppl_accum / n_real_batches if n_real_batches > 0 else 0.0

        print(f"📊 Results:")
        print(f"   > Real Data PPL (Oracle): {avg_real_ppl:.4f}")
        print(f"   > Diffusion PPL (Yours):  {avg_gen_ppl:.4f}")
        print(f"   > Gap:                    {avg_gen_ppl - avg_real_ppl:.4f}")
        
        diffusion_model.train()
        return avg_gen_ppl, avg_real_ppl