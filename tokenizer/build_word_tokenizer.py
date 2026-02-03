from pathlib import Path
import json

# paths
data_path = Path("data/cleaned/balti_clean.txt")
tokenizer_path = Path("tokenizer/word_tokenizer.json")

tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

# read data
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# split into words
words = text.split()
unique_words = sorted(list(set(words)))

# add special tokens
special_tokens = ["<unk>", "<pad>", "<eos>"]
vocab = special_tokens + unique_words

# create mappings
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

tokenizer = {
    "vocab_size": len(vocab),
    "stoi": stoi,
    "itos": itos
}

# save tokenizer
with open(tokenizer_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, ensure_ascii=False, indent=2)

print(f"Word tokenizer built successfully!")
print(f"Vocab size: {len(vocab)}")
print(f"Special tokens: {special_tokens}")
print(f"Sample words: {unique_words[:10]}")