import torch
from torch.utils.data import Dataset

class CorticoDataset(Dataset):
    def __init__(self, data_tensor, seq_length):

        if not isinstance(data_tensor, torch.Tensor):
            self.data = torch.tensor(data_tensor).long()
        else:
            self.data = data_tensor.long()
            
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        
        return x, y