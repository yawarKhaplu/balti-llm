from pathlib import Path
import json

# paths
data_path = Path("data/cleaned/balti_clean.txt")
tokenizer_path = Path("tokenizer/tokenizer.json")

tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

# read data
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# get unique characters
chars = sorted(list(set(text)))

# create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

tokenizer = {
    "vocab_size": len(chars),
    "stoi": stoi,
    "itos": itos
}

# save tokenizer
with open(tokenizer_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, ensure_ascii=False, indent=2)

print(f"Tokenizer built successfully!")
print(f"Vocab size: {len(chars)}")
print(f"Characters: {chars}")
