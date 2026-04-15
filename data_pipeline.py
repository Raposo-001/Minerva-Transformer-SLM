import torch
import re
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text) 
    return text.split()

def build_vocab(tokens):
    counts = Counter(tokens)
    vocab = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"] + [w for w, c in counts.items() if c > 1]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word

with open("book.txt", "r", encoding="utf-8") as f:
    book = f.read()

cleanbook = preprocess_text(book)
finalbook = build_vocab(cleanbook)