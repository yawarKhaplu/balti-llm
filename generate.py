import torch
from tokenizer.tokenizer import CharTokenizer
from model.minigpt import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CharTokenizer()

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=32,
).to(device)

# load trained weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

def generate(start_text, max_new_chars=100, temperature=0.8, top_k=10):
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)

    for _ in range(max_new_chars):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, ix, v)
            logits = mask

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


print(generate("xaan", 100, temperature=1.2, top_k=5))


