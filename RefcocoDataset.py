import torch
from torch.utils.data import Dataset
from const import global_consts as gc

class LMDataset(Dataset):
    def __init__(self, label):
        def padding(sen, pad):
            if len(sen) >= pad:
                return sen[:pad]
            else:
                return sen[:] + [gc.PAD_id for i in range(len(sen), pad)]

        def trunc(sen, mlen):
            if len(sen) >= mlen:
                return sen[:mlen]
            else:
                return sen[:]

        self.input = []
        self.output = []
        self.length = []
        for sen in label:
            self.input.append(padding(sen, gc.padding_len))
            self.length.append(min(gc.max_len, len(sen)))
            self.output.append(padding(sen, gc.max_len))
        self.num = len(self.input)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        length = len(self.output[idx])
        return torch.tensor(self.input[idx]), torch.tensor(self.output[idx]), self.length[idx]
#        return torch.cumsum(torch.ones_like(torch.tensor(self.input[idx]), dtype=torch.long), dim=0), \
#                torch.cumsum(torch.ones_like(torch.tensor(self.output[idx]), dtype=torch.long), dim=0), self.length[idx]
