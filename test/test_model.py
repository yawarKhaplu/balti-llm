import torch
from tokenizer.word_tokenizer import WordTokenizer
from model.minigpt import MiniGPT

tokenizer = WordTokenizer()

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=32,
)

# fake input (batch=1, sequence=10)
idx = torch.randint(0, tokenizer.vocab_size, (1, 10))

logits, loss = model(idx)

print("Logits shape:", logits.shape)
print("Loss:", loss)
