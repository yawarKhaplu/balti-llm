import json
from pathlib import Path

class WordTokenizer:
    def __init__(self, tokenizer_path="tokenizer/word_tokenizer.json"):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = data["vocab_size"]
        self.unk_token = "<unk>"

    def encode(self, text):
        words = text.split()
        return [self.stoi.get(word, self.stoi[self.unk_token]) for word in words]

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])