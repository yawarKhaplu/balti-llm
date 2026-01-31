from tokenizer.tokenizer import CharTokenizer

tokenizer = CharTokenizer()

text = "nga balti zaban"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Original:", text)
print("Encoded:", ids)
print("Decoded:", decoded)
