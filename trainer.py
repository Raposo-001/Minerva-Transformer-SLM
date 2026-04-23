from model_arch import Transformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline import data_tensor, word2idx
from dataset_utils import CorticoDataset 


dataset = CorticoDataset(data_tensor, seq_length=200)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

model = Transformer(vocab_size=len(word2idx), d_model=256, nhead=8, num_layers=6)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
vocab_size = len(word2idx)

for epoch in range(1000): 
    model.train()
    total_loss = 0
    
    for batch in dataloader: 
        optimizer.zero_grad() 
        
        output = model(batch.input)
        loss = criterion(output.view(-1, vocab_size), batch.target.view(-1))
        
        loss.backward()
        optimizer.step() 
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} finalizada. Erro médio: {total_loss/len(dataloader)}")