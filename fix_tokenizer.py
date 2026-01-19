import json
from pathlib import Path

# Config
FILE_PATH = Path("datasets/wikitext-103/tokenizer_wiki_65k.json")
BACKUP_PATH = Path("datasets/wikitext-103/tokenizer_wiki_65k.json.bak")

def fix_tokenizer_format():
    if not FILE_PATH.exists():
        print(f"❌ File not found: {FILE_PATH}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if 'model' and 'merges' exist
    if "model" not in data or "merges" not in data["model"]:
        print("⚠️  JSON structure unexpected (missing model or merges keys).")
        return

    merges = data["model"]["merges"]
    
    # Check if conversion is actually needed
    if not merges:
        print("⚠️  Merges list is empty.")
        return

    if isinstance(merges[0], str):
        print("✅ Merges are already in string format. No changes needed.")
        return

    if isinstance(merges[0], list):
        print(f"🔧 Detected list-format merges (e.g., {merges[0]}). Converting to strings...")
        
        # Create backup
        print(f"💾 Backing up original to {BACKUP_PATH}...")
        with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # CONVERSION LOGIC: Join ["a", "b"] -> "a b"
        new_merges = [" ".join(pair) for pair in merges]
        data["model"]["merges"] = new_merges
        
        # Save overwrite
        print(f"💾 Saving fixed JSON to {FILE_PATH}...")
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print("✅ Success! The tokenizer should now load correctly in train_ar.py")
    else:
        print(f"❌ Unknown merge format: {type(merges[0])}. Cannot fix automatically.")

if __name__ == "__main__":
    fix_tokenizer_format()