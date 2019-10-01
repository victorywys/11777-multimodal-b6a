import torch
from torch.utils.data import Dataset
from const import global_consts as gc

class RefcocoDataset(Dataset):
    def __init__(self, img, label):
        self.input = img
        self.output = label
        if (len(self.input) != len(self.output)):
            print("Warning: the number of images(%d) is not equal to the number of labels(%d)")
        self.num = len(self.input)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = torch.tensor(self.output[idx])
        if len(self.output[idx]) >= gc.max_len:
            label = torch.tensor(self.output[idx][:gc.max_len])
            length = gc.max_len
        else:
            label = torch.cat([torch.tensor(self.output[idx]), torch.zeros(gc.max_len - len(self.output[idx]), dtype=torch.long)], 0)
            length = len(self.output[idx])
        return self.input[idx], label, length
