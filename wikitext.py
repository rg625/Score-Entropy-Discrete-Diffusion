from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from ml_collections import config_dict

# --------------------------------------------------------------------
# 1. Bit Manipulation Helpers  (UNCHANGED API)
# --------------------------------------------------------------------

# Vocab is exactly 65,536 (2^16)
BITS_PER_TOKEN = 16

def int_to_gray(n: int) -> int:
    """Convert integer to Gray code."""
    return n ^ (n >> 1)

def gray_to_int(n: int) -> int:
    """Convert Gray code to integer."""
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

def clean_wikitext_artifacts(text: str) -> str:
    """
    Post-processing to make WikiText-103 human-readable.
    Removes tokenization artifacts (@-@, @,@) and fixes spacing.
    """
    # 1. Replace WikiText specific artifacts
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")

    # 2. Fix punctuation spacing
    text = re.sub(r"\s+([,.:;?!%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = text.replace(" ' ", "'")

    return text.strip()

def bits_to_text_semantic(bits: torch.Tensor, tokenizer: Any, new_to_old_map: Dict[int, int]) -> str:
    """
    Decodes bits back to text using the Semantic Gray Code mapping.
    Crucial for callbacks/visualization.

    Args:
        bits: Flat tensor of bits [SeqLen * 16]
        tokenizer: Tokenizer instance (from tokenizers library)
        new_to_old_map: Dict mapping Rank -> Original Token ID
    """
    bits = bits.to(torch.long).view(-1)
    S = bits.numel()

    # Truncate to ensure multiple of BITS_PER_TOKEN
    if S % BITS_PER_TOKEN != 0:
        S = (S // BITS_PER_TOKEN) * BITS_PER_TOKEN
        bits = bits[:S]

    token_ids: List[int] = []

    # Process chunks of 16 bits
    for i in range(0, S, BITS_PER_TOKEN):
        chunk = bits[i : i + BITS_PER_TOKEN]

        # Bits -> Integer (Gray Value)
        gray_val = 0
        for b in chunk.tolist():
            gray_val = (gray_val << 1) | int(b)

        # Gray Value -> Rank Index
        rank_idx = gray_to_int(gray_val)

        # Rank Index -> Original Token ID
        # With our 16-bit vocab (65,536), this map covers ALL possible ranks.
        if rank_idx in new_to_old_map:
            token_ids.append(new_to_old_map[rank_idx])
        else:
            # This branch is theoretically unreachable with a 2^16 vocab
            pass

    # Decode using the custom BPE tokenizer
    return tokenizer.decode(token_ids)

def batch_bits_to_text_semantic(bits_batch: torch.Tensor, tokenizer: Any, map_dict: Dict[int, int]) -> List[str]:
    """Helper to decode a whole batch of sequences."""
    return [bits_to_text_semantic(bits_batch[i], tokenizer, map_dict) for i in range(bits_batch.size(0))]

def load_wikitext_semantic_assets(root: Path = Path("./datasets/wikitext-103")):
    from tokenizers import Tokenizer
    tok_path = root / "tokenizer_wiki_65k.json"
    map_path = root / "semantic_mapping_wiki_65k.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    with open(map_path, "r") as f:
        maps = json.load(f)
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old

# --------------------------------------------------------------------
# 2. Unified WikiTextDataset (OLD semantic-bits + NEW token mode)
# --------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """
    WikiText-103 dataset supporting BOTH:

    (OLD) Semantic Binary mode:
        cfg.data.representation == "binary" AND cfg.data.binarization == "semantic"
        __getitem__ returns FloatTensor [seq_len_tokens * 16]
        Uses semantic_mapping_wiki_65k.json and token_to_bits_table.

    (NEW) Token mode (for autoregressive training):
        cfg.data.representation == "tokens"
        __getitem__ returns LongTensor [seq_len_tokens]
        Does NOT require semantic map, and does NOT build bit lookup table.

    Notes:
      - Tokenization uses tokenizer_wiki_65k.json in BOTH modes.
      - Caches are separate to avoid accidental mixing.
    """

    is_text_dataset = True  # Signal for callbacks

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.config = config
        self.split = split

        # --------- Mode selection (default preserves old behavior) ----------
        self.representation = str(getattr(config.data, "representation", "binary")).lower()
        self.binarization = str(getattr(config.data, "binarization", "semantic")).lower()

        self.token_mode = (self.representation == "tokens")
        self.semantic_binary_mode = (self.representation == "binary" and self.binarization == "semantic")

        if not (self.token_mode or self.semantic_binary_mode):
            raise ValueError(
                f"Unsupported WikiText mode: representation={self.representation}, binarization={self.binarization}. "
                f"Supported: tokens OR (binary+semantic)."
            )

        # --------- PATH CONFIGURATION (defaults identical to old) ----------
        self.root = Path(getattr(config.data, "root", "./datasets/wikitext-103"))

        # Point to the 16-bit files
        map_path = Path(getattr(config.data, "semantic_map_path", self.root / "semantic_mapping_wiki_65k.json"))
        tokenizer_path = Path(getattr(config.data, "tokenizer_path", self.root / "tokenizer_wiki_65k.json"))

        # Select filenames (identical to old)
        if split == "train":
            raw_filename = "wiki.train.tokens"
        elif split == "val":
            raw_filename = "wiki.valid.tokens"
        else:
            raw_filename = "wiki.test.tokens"
        raw_path = self.root / raw_filename

        # OLD cache name (keep it for backward compatibility in semantic-binary mode)
        old_cache_path = self.root / f"cached_{split}_wiki_65k.pt"
        # NEW cache name for token-mode to avoid collisions
        token_cache_path = self.root / f"cached_{split}_wiki_65k_token_ids.pt"

        # 1. Load Tokenizer (required in BOTH modes)
        # -----------------------------------------
        if not tokenizer_path.exists():
            raise RuntimeError(f"❌ Tokenizer not found at {tokenizer_path}. Run scripts/1_train_wiki_bpe.py")

        print(f"[{split.upper()}] Loading Tokenizer: {tokenizer_path}")
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Sanity check
        if self.vocab_size != 65536:
            print(f"⚠️ Warning: Tokenizer vocab size is {self.vocab_size}, expected 65536.")

        # 2. Load Semantic Map + Build Lookup Table ONLY for old mode
        # ----------------------------------------------------------
        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: Dict[int, int] = {}
        self.token_to_bits_table: Optional[torch.Tensor] = None

        if self.semantic_binary_mode:
            if not map_path.exists():
                raise RuntimeError(f"❌ Semantic map not found at {map_path}. Run scripts/2_prepare_wiki_semantic_map.py")

            print(f"[{split.upper()}] Loading Semantic Map...")
            with open(map_path, "r") as f:
                map_data = json.load(f)
                self.old_to_new = {int(k): int(v) for k, v in map_data["old_to_new"].items()}
                self.new_to_old = {int(k): int(v) for k, v in map_data["new_to_old"].items()}

            # Build Lookup Table [Vocab, 16] (identical behavior to old)
            print(f"[{split.upper()}] Building token->bits lookup table ...")
            table = torch.zeros((self.vocab_size, BITS_PER_TOKEN), dtype=torch.float32)
            for tid in range(self.vocab_size):
                rank = self.old_to_new.get(tid, 0)
                gray_val = int_to_gray(rank)
                for b in range(BITS_PER_TOKEN):
                    shift = BITS_PER_TOKEN - 1 - b
                    bit = (gray_val >> shift) & 1
                    table[tid, b] = float(bit)
            self.token_to_bits_table = table

        # 3. Load Data (With Caching)
        # ---------------------------
        # IMPORTANT: In BOTH modes we ultimately need token IDs.
        # For backward compatibility:
        #   - semantic-binary mode continues to read/write old_cache_path
        #   - token mode uses token_cache_path
        cache_path = old_cache_path if self.semantic_binary_mode else token_cache_path

        if cache_path.exists():
            print(f"[{split.upper()}] Loading cached tensor from {cache_path}...")
            self.data_tokens = torch.load(cache_path)
        else:
            if not raw_path.exists():
                raise RuntimeError(f"❌ Raw file {raw_path} not found.")

            print(f"[{split.upper()}] Tokenizing raw text (First run only)...")
            with open(raw_path, "r", encoding="utf-8") as f:
                text = f.read()

            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids

            self.data_tokens = torch.tensor(tokens, dtype=torch.long)
            torch.save(self.data_tokens, cache_path)
            print(f"[{split.upper()}] Cached processed tokens to {cache_path}")

        # 4. Sequence Logic (identical fields)
        # -----------------------------------
        self.seq_len_tokens = int(getattr(config.data, "sequence_len_tokens", 1024))
        self.seq_len_bits = self.seq_len_tokens * BITS_PER_TOKEN
        self.num_sequences = len(self.data_tokens) // self.seq_len_tokens

        mode_str = "TOKENS" if self.token_mode else "BINARY/SEMANTIC"
        print(f"[{split.upper()}] Ready ({mode_str}). {len(self.data_tokens)} tokens -> {self.num_sequences} sequences.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 1. Get Token Slice
        start = idx * self.seq_len_tokens
        end = start + self.seq_len_tokens
        tokens = self.data_tokens[start:end]  # [SeqLen] (long)

        # NEW: Token mode returns token IDs
        if self.token_mode:
            return tokens

        # OLD: Semantic binary mode returns bits (identical behavior)
        if self.semantic_binary_mode:
            if self.token_to_bits_table is None:
                raise RuntimeError("token_to_bits_table missing in semantic binary mode.")
            bits = self.token_to_bits_table[tokens]   # [SeqLen,16]
            return bits.view(-1)                      # [SeqLen*16]

        # Should be unreachable due to checks in __init__
        raise RuntimeError("Unsupported mode reached in __getitem__.")

# --------------------------------------------------------------------
# 3. Dataloader Factory (same signature; token-mode friendly defaults)
# --------------------------------------------------------------------

def get_dataloaders(
    config: config_dict.ConfigDict,
    *,
    batch_size: int | None = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train / val / test loaders for WikiText-103.

    Preserves old defaults for semantic-binary mode:
      train drop_last=True, val drop_last=True, test drop_last=False

    Token mode (AR) usually can keep val/test complete, but we keep old defaults
    unless overridden by cfg.data.drop_last_val / cfg.data.drop_last_train.
    """
    batch = batch_size or config.train.batch_size

    num_workers = int(getattr(config.data, "num_workers", 4))
    prefetch_factor = int(getattr(config.data, "prefetch_factor", 2))
    pin_memory = bool(getattr(config.data, "pin_memory", True))
    persistent_workers = num_workers > 0

    # allow overrides, but default exactly as before
    drop_last_train = bool(getattr(config.data, "drop_last_train", True))
    drop_last_val   = bool(getattr(config.data, "drop_last_val", True))

    def make_loader(ds: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    print("\n--- Initializing WikiText Datasets ---")
    train_ds = WikiTextDataset(config, split="train")
    val_ds   = WikiTextDataset(config, split="val")
    test_ds  = WikiTextDataset(config, split="test")

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=drop_last_train)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=drop_last_val)
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

# --------------------------------------------------------------------
# 4. Sanity Check (preserves old test, adds token-mode test)
# --------------------------------------------------------------------
if __name__ == "__main__":
    class Config:
        class data:
            sequence_len_tokens = 512
            num_workers = 2
            prefetch_factor = 2
            pin_memory = False
            # toggle:
            representation = "binary"
            binarization = "semantic"
        class train:
            batch_size = 4

    cfg = Config()

    print("Testing WikiTextDataset (binary/semantic)...")
    try:
        tr, va, te = get_dataloaders(cfg)
        batch = next(iter(tr))
        print(f"Batch Shape: {batch.shape}")  # [B, 512*16]
        ds = tr.dataset
        decoded_text = bits_to_text_semantic(batch[0], ds.tokenizer, ds.new_to_old)
        print(f"Decoded Sample (First 100 chars): {decoded_text[:100]}...")
        print("✅ Binary/semantic test passed.")
    except Exception as e:
        print(f"❌ Binary/semantic test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting WikiTextDataset (tokens)...")
    try:
        cfg.data.representation = "tokens"
        tr, va, te = get_dataloaders(cfg)
        batch = next(iter(tr))
        print(f"Batch Shape: {batch.shape}")  # [B, 512]
        ds = tr.dataset
        decoded = ds.tokenizer.decode(batch[0].tolist())
        print(f"Decoded Sample (First 100 chars): {decoded[:100]}...")
        print("✅ Token-mode test passed.")
    except Exception as e:
        print(f"❌ Token-mode test failed: {e}")
        import traceback
        traceback.print_exc()
