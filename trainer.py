from model_arch import Transformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline import data_tensor, word2idx

class CorticoDataset(Dataset):
    def __init__(self, data_tensor, seq_length):
        self.data = data_tensor
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


SEQ_LENGTH = 100 
dataset = CorticoDataset(data_tensor, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Transformer(vocab_size=len(word2idx), d_model=256, nhead=8, num_layers=6)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000): # Grande Loop (Epochs)
    model.train()
    total_loss = 0
    
    for batch in dataloader: # Pequeno Loop (Batches)
        optimizer.zero_grad() # Limpa a memória do erro anterior
        
        output = model(batch.input) # A IA tenta adivinhar (Forward)
        
        # O erro é calculado comparando o que ela disse com o que o livro diz
        loss = criterion(output.view(-1, vocab_size), batch.target.view(-1))
        
        loss.backward() # Backpropagation: o erro volta "avisando" o que está errado
        optimizer.step() # O ajuste real dos pesos (a IA aprende)
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} finalizada. Erro médio: {total_loss/len(dataloader)}")