import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
except ImportError:
    print("Error: Library 'tokenizers' missing. Install via: pip install tokenizers")
    sys.exit(1)

# Added datasets import to robustly fetch the data
try:
    from datasets import load_dataset
except ImportError:
    print("Error: Library 'datasets' missing. Install via: pip install datasets")
    sys.exit(1)


def prepare_text8_data(root):
    """
    Downloads text8 via huggingface datasets and saves the training split 
    to a text file for tokenizer training.
    """
    raw_path = root / "text8_train_raw.txt"
    
    # If already prepared, return path
    if raw_path.exists():
        print(f"Found existing raw data at: {raw_path}")
        return raw_path

    print("Downloading text8 dataset (afmck/text8)...")
    # Using the same dataset reference as data.py
    # This automatically handles caching and downloading
    dataset = load_dataset("afmck/text8", split="train")
    
    print(f"Writing training data to {raw_path}...")
    with open(raw_path, "w", encoding="utf-8") as f:
        # Iterate through the dataset and write raw text to file
        for example in dataset:
            text = example.get('text', '')
            if text:
                f.write(text + "\n")
    
    return raw_path


def train_text8_bpe():
    root = Path("./datasets/text8")
    os.makedirs(root, exist_ok=True)
    
    # Updated: Download/Prepare data using HF datasets instead of custom import
    raw_path = prepare_text8_data(root)
    save_path = root / "tokenizer_bpe_16k.json"

    print(f"--- Training BPE Tokenizer on {raw_path} ---")
    print("Strategy: Metaspace (Space -> '_')")

    # 1. Initialize Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # COMPATIBILITY FIX: Removed 'add_prefix_space=True' to support older versions.
    # Metaspace replaces " " with "_" and usually handles the prefix by default.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="_")
    
    # Decoder reverses this: "_" -> " "
    tokenizer.decoder = decoders.Metaspace(replacement="_")

    # 2. Configure Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        min_frequency=2,
        show_progress=True,
        # IMPORTANT: Added [EOS] so data.py can use tokenizer.eos_token
        special_tokens=["[UNK]", "[PAD]", "[EOS]"],
        # Add "_" to the alphabet so the tokenizer can build up from it
        initial_alphabet=list("abcdefghijklmnopqrstuvwxyz_") 
    )

    # 3. Train
    tokenizer.train([str(raw_path)], trainer)

    # 4. Save
    tokenizer.save(str(save_path))
    
    print("-" * 40)
    print(f"✅ SUCCESS: Tokenizer saved to: {save_path}")
    print("The run_train.py script will now detect this automatically if config.data.train == 'text8'")
    print("-" * 40)

if __name__ == "__main__":
    train_text8_bpe()