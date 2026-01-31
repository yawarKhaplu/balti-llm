import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, tokenizer_path="tokenizer/tokenizer.json"):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = data["vocab_size"]

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])
