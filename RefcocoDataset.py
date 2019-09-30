from torch.utils.data.dataset import Dataset

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
        return self.input[idx], self.output[idx]
