import torch
from tokenizer.word_tokenizer import WordTokenizer
from model.minigpt import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = WordTokenizer()
model = MiniGPT(vocab_size=tokenizer.vocab_size, block_size=32).to(device)

# Load trained weights
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

def generate(start_text, max_new_words=10, temperature=0.8, top_k=10):
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)
    
    for _ in range(max_new_words):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, ix = torch.topk(logits, min(top_k, logits.shape[-1]))
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, ix, v)
            logits = mask
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    
    return tokenizer.decode(idx[0].tolist())

# Test with Balti prompts
test_prompts = ["slam", "na", "yang", "balti", "khaplu"]

while True:
    input_text = input("Enter starting text (or 'exit' to quit): ")
    if input_text.lower() == 'exit':
        break

    print(generate(input_text, 20, temperature=1.2, top_k=5))