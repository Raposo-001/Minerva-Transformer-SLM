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
word2idx, idx2word = finalbook
input_ids = [word2idx.get(word, word2idx["<UNK>"]) for word in cleanbook]
data_tensor = torch.tensor(input_ids, dtype=torch.long)


save_dict = {
    "tokens": data_tensor,
    "word2idx": word2idx,
    "idx2word": idx2word
}


torch.save(save_dict, "dados_preparados.pt")

print("--- Pipeline Finalizado ---")
print(f"Total de palavras no livro: {len(cleanbook)}")
print(f"Tamanho do vocabulário: {len(word2idx)}")
print("Arquivo 'dados_preparados.pt' gerado com sucesso!")