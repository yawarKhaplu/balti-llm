from pathlib import Path

# paths
raw_path = Path("data/raw/balti_raw.txt")
clean_path = Path("data/cleaned/balti_clean.txt")

# make sure cleaned folder exists
clean_path.parent.mkdir(parents=True, exist_ok=True)

# English words to filter out
english_words = {
''}

sentences = set()

with open(raw_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().lower()
        if line:
            # Filter out sentences with English words
            words = line.split()
            if not any(word in english_words for word in words):
                sentences.add(line)

with open(clean_path, "w", encoding="utf-8") as f:
    for s in sorted(sentences):
        f.write(s + "\n")

print(f"Raw lines: {sum(1 for _ in open(raw_path, 'r', encoding='utf-8'))}")
print(f"Clean lines: {len(sentences)}")
