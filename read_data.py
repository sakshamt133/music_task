from torch.utils.data import Dataset
from read_dataset import df
import numpy as np
import torch


class Music(Dataset):
    def __init__(self):
        super(Music, self).__init__()
        all_data = np.array(df)
        x_train = all_data[:, 0:2]
        self.x_train = torch.from_numpy(x_train)

        y_train = all_data[:, -1]
        self.y_train = torch.from_numpy(y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

    def __len__(self):
        return self.x_train.shape[0]
