import os
import shutil
import tarfile
import subprocess
import csv  # <--- Using built-in csv instead of pandas
from pathlib import Path

# --- Configuration ---
DATASET_DIR = Path("datasets/wikitext-103")
TGZ_FILENAME = "wikitext-103.tgz"
URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"

def system_download(url, dest_path):
    print(f"⬇️  Downloading {url}...")
    if shutil.which("wget"):
        subprocess.run(["wget", "-O", str(dest_path), url], check=True)
    elif shutil.which("curl"):
        subprocess.run(["curl", "-L", "-o", str(dest_path), url], check=True)
    else:
        raise EnvironmentError("Neither wget nor curl found.")

def read_lines(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]

def save_data(lines, raw_path, csv_path):
    # 1. Save Raw
    with open(raw_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    
    # 2. Save CSV (Standard Python version)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text'])
        for line in lines:
            writer.writerow([line])

    print(f"   ✅ Saved {raw_path.name} and {csv_path.name} ({len(lines)} lines)")

def prepare_wikitext():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = DATASET_DIR / TGZ_FILENAME
    
    # Download
    if not tgz_path.exists() or tgz_path.stat().st_size < 100_000_000:
        system_download(URL, tgz_path)
    
    # Extract
    print("\n🔍 Extracting Archive...")
    extract_temp = DATASET_DIR / "temp_extract"
    if extract_temp.exists(): shutil.rmtree(extract_temp)
    extract_temp.mkdir()

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(extract_temp)
    
    # Identify
    all_files = list(extract_temp.rglob("*"))
    train_src, valid_src, test_src = None, None, None

    for f in all_files:
        if not f.is_file(): continue
        name = f.name.lower()
        if "valid" in name or "dev" in name: valid_src = f
        elif "test" in name: test_src = f
        elif "train" in name: train_src = f

    if train_src is None and all_files:
        files_with_size = sorted([(f, f.stat().st_size) for f in all_files if f.is_file()], key=lambda x: x[1], reverse=True)
        if files_with_size: train_src = files_with_size[0][0]

    if not train_src: raise RuntimeError("❌ Could not find a training file.")

    # Process
    print(f"\n⚙️  Processing Data from {train_src.name}...")
    train_lines = read_lines(train_src)
    
    if valid_src:
        valid_lines = read_lines(valid_src)
    else:
        print("   ⚠️  Splitting Validation from Train...")
        split_idx = int(len(train_lines) * 0.95)
        valid_lines = train_lines[split_idx:]
        train_lines = train_lines[:split_idx]

    test_lines = read_lines(test_src) if test_src else valid_lines

    # Save
    print("\n💾 Saving Cleaned Files...")
    save_data(train_lines, DATASET_DIR / "wiki.train.tokens", DATASET_DIR / "train.csv")
    save_data(valid_lines, DATASET_DIR / "wiki.valid.tokens", DATASET_DIR / "valid.csv")
    save_data(test_lines,  DATASET_DIR / "wiki.test.tokens",  DATASET_DIR / "test.csv")

    shutil.rmtree(extract_temp)
    print("\n✨ Done.")

if __name__ == "__main__":
    prepare_wikitext()