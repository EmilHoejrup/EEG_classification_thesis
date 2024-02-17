import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as p
import gzip
from pathlib import Path
from support.constants import *
from sklearn.preprocessing import StandardScaler

X_HIM_OR_HER = DATA_DIR / 'him-or-her' / 'X.npy.gz'
Y_HIM_OR_HER = DATA_DIR / 'him-or-her' / 'Y.p'


class HimOrHer(Dataset):
    def __init__(self, train=True, val_split=0.2):
        self.train = train
        self.X_file = X_HIM_OR_HER
        self.Y_file = Y_HIM_OR_HER

        with gzip.GzipFile(self.X_file, "r") as f:
            self.X = torch.from_numpy(np.load(file=f))

        with open(self.Y_file, 'rb') as f:
            self.Y = p.load(f)
        # Convert to float 32
        self.X = self.X.type(torch.float32)
        self.Y = self.Y.type(torch.float32)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=val_split, random_state=43)
        self.X_train = self.z_score_normalization(self.X_train)
        self.X_val = self.z_score_normalization(self.X_val)

    # Applies Z-score normalization to each channel individually
    def z_score_normalization(self, tensor, axis=2):
        mean = torch.mean(tensor, dim=axis, keepdim=True)
        std = torch.std(tensor, dim=axis, keepdim=True)
        # Adding small number to avoid division by zero
        tensor = (tensor - mean) / (std + 1e-8)
        return tensor

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_val)

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.Y_train[index]
        else:
            return self.X_val[index], self.Y_val[index]
