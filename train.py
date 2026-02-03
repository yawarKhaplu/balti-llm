import torch
from tokenizer.word_tokenizer import WordTokenizer
from model.minigpt import MiniGPT

# hyperparameters
block_size = 32
batch_size = 8
max_steps = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer
tokenizer = WordTokenizer()

# load data
with open("data/cleaned/balti_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# model
model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# training loop
for step in range(max_steps):
    x, y = get_batch()
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.4f}")
torch.save(model.state_dict(), "model.pth")
