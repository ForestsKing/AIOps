import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, w, num_class):
        self.X = np.eye(num_class)[data]
        self.y = data
        self.w = w

    def __getitem__(self, index):
        X = self.X[index:index + self.w]
        y = self.y[index + self.w]

        return X, y

    def __len__(self):
        return len(self.X) - self.w
