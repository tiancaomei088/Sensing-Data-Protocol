import torch
import numpy as np

from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, data_list, labels):
        data_list = np.abs(data_list)
        self.data_list = torch.from_numpy(data_list).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx], self.labels[idx]