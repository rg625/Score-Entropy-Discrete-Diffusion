import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
from ml_collections import config_dict
import numpy as np
import wandb  # Highly recommended for tracking AR curves

# Import your dataset class
# Ensure your dataset file provided in the prompt is named 'dataset_wiki.py'
from wikitext import WikiTextDataset

def get_ar_config():
    """
    Configuration strictly aligned with the Discrete Diffusion setup.
    """
    cfg = config_dict.ConfigDict()
    
    # Data Config
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = "./datasets/wikitext-103"
    cfg.data.representation = "tokens"     # <--- Crucial: Train on tokens, not bits
    cfg.data.sequence_len_tokens = 512     # Context window (must match or exceed diffusion model)
    cfg.data.batch_size = 16               # Adjust based on VRAM
    cfg.data.num_workers = 4
    cfg.data.prefetch_factor = 2
    cfg.data.pin_memory = True
    cfg.data.drop_last_train = True
    cfg.data.drop_last_val = True

    # Model Config (GPT-2 Small equivalent, customized for 65k vocab)
    cfg.model = config_dict.ConfigDict()
    cfg.model.n_layer = 12
    cfg.model.n_head = 12
    cfg.model.n_embd = 768
    cfg.model.vocab_size = 65536           # Matches tokenizer_wiki_65k.json
    
    # Training Config
    cfg.train = config_dict.ConfigDict()
    cfg.train.lr = 3e-4
    cfg.train.weight_decay = 0.01
    cfg.train.epochs = 20
    cfg.train.grad_clip = 1.0
    cfg.train.save_dir = "./checkpoints_ar"
    cfg.train.wandb_project = "ar-baseline-wikitext"
    
    return cfg

def train_autoregressive():
    cfg = get_ar_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    print(f"🚀 Starting AR Training on {device}")
    
    # --- 1. Dataset Setup ---
    print("Loading Datasets...")
    # These initialize in 'tokens' mode because of cfg.data.representation
    train_ds = WikiTextDataset(cfg, split="train")
    valid_ds = WikiTextDataset(cfg, split="val")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last_train
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=cfg.data.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last_val
    )

    # --- 2. Model Setup ---
    print(f"Initializing GPT-2 (Vocab: {cfg.model.vocab_size})...")
    model_config = GPT2Config(
        vocab_size=cfg.model.vocab_size,
        n_positions=cfg.data.sequence_len_tokens,
        n_ctx=cfg.data.sequence_len_tokens,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        bos_token_id=None, 
        eos_token_id=None,
        use_cache=False 
    )
    model = GPT2LMHeadModel(model_config).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # Optional: WandB
    use_wandb = False
    if cfg.train.wandb_project:
        try:
            wandb.init(project=cfg.train.wandb_project, config=cfg.to_dict())
            use_wandb = True
        except:
            print("WandB not detected, skipping logging.")

    # --- 3. Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(cfg.train.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch in pbar:
            # batch is [B, SeqLen] LongTensor
            inputs = batch.to(device)
            
            # GPT2LMHeadModel calculates CrossEntropy automatically if labels provided
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if use_wandb:
                wandb.log({"train/loss": loss.item()})

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        steps = 0
        print("Running Validation...")
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch.to(device)
                outputs = model(inputs, labels=inputs)
                val_loss_accum += outputs.loss.item()
                steps += 1
        
        avg_val_loss = val_loss_accum / steps
        val_ppl = np.exp(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.4f}")
        
        if use_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/ppl": val_ppl,
                "epoch": epoch + 1
            })

        # --- Checkpointing ---
        # Save latest
        torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, "ar_last.pt"))
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(cfg.train.save_dir, "ar_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New Best PPL: {val_ppl:.4f}. Saved to {best_path}")

    print("Training Complete.")

if __name__ == "__main__":
    train_autoregressive()