import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as p
import gzip
from pathlib import Path
from itertools import product
from sklearn.cluster import KMeans
from support.constants import *
import yaml
from sklearn.preprocessing import StandardScaler
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
from support.utils import *
from braindecode.datautil import load_concat_dataset

X_HIM_OR_HER = DATA_DIR / 'him-or-her' / 'X.npy.gz'
Y_HIM_OR_HER = DATA_DIR / 'him-or-her' / 'Y.p'
BNCI2014_001_DIR = DATA_DIR / 'BNCI2014_001'
X_BNCI2014_001 = BNCI2014_001_DIR / 'X.npy.gz'
Y_BNCI2014_001 = BNCI2014_001_DIR / 'Y.p'
BCI_IV_2A_DIR = DATA_DIR / 'BCI_IV_2A'
X_BCI_IV_2A = BCI_IV_2A_DIR / 'X.npy.gz'
Y_BCI_IV_2A = BCI_IV_2A_DIR / 'Y.p'


def read_X_and_y(x_file, y_file):
    with gzip.GzipFile(x_file, "r") as f:
        X = torch.from_numpy(np.load(file=f))

    with open(y_file, 'rb') as f:
        Y = p.load(f)
    return X, Y


def fetch_data(data_dir):
    if data_dir == BCI_IV_2A_DIR:
        if not BCI_IV_2A_DIR.exists():
            fetch_BCI_IV_2A()
        X, Y = read_X_and_y(X_BCI_IV_2A, Y_BCI_IV_2A)
    elif data_dir == BNCI2014_001_DIR:
        if not BNCI2014_001_DIR.exists():
            fetch_BNCI2014_001()
        X, Y = read_X_and_y(X_BNCI2014_001, Y_BNCI2014_001)
    return X, Y


class CONTINUOUS_DATASET(Dataset):
    def __init__(self, data_dir, train=True):
        self.X, self.Y = fetch_data(data_dir)
        self.train = train

        self.X = self.X.to(torch.float32)
        _, _, timepoints = self.X.shape
        baseline_start = 0
        baseline_end = 125
        baseline_mean = torch.mean(
            self.X[:, :, baseline_start:baseline_end], dim=2, keepdim=True)
        self.X = self.X - baseline_mean
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)

    def get_X_shape(self):
        return self.X_train.shape

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


class BCI_2A_CONTINUOUS(CONTINUOUS_DATASET):
    def __init__(self, train=True):
        super().__init__(BCI_IV_2A_DIR, train)


class BCI_2B_CONTINUOUS(CONTINUOUS_DATASET):
    def __init__(self, train=True):
        super().__init__(BCI_IV_2B_DIR, train)


class PERMUTATION_DATASET(Dataset):
    def __init__(self, window_size, stride, data_dir, train=True, threshold=0.001, ):
        self.X, self.Y = fetch_data(data_dir)

        self.train = train

        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold

        _, _, timepoints = self.X.shape

        self.X = F.max_pool1d(self.X, 3, 3)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=43)
        if self.train:
            self.X_train = self._discretize(self.X_train)
            # self.X_train = self.X_train.to(torch.float32)
        else:
            self.X_val = self._discretize(self.X_val)
            # self.X_val = self.X_val.to(torch.float32)

    def get_X_shape(self):
        return self.X_train.shape

    def get_vocab_size(self):
        return len(self.permutations)

    def _discretize(self, X):
        # self.p_length = configs['BNCI2014_001']['window_size']
        values = list(range(self.window_size))

        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.window_size)]
        images, channels, timepoints = X.shape
        discretized_X = []
        for image in range(images):
            image_sequence = []
            for channel in range(channels):
                sequence = X[image, channel]
                discretized_sequence = self._permute(sequence)
                image_sequence.append(discretized_sequence)
            discretized_X.append(image_sequence)
        return torch.tensor(discretized_X, dtype=torch.long)

    def _permute(self, X):
        X = X.tolist()
        new_sequence = []

        for i in range(0, len(X) - self.window_size, self.stride):
            window = X[i:(i+self.window_size)]
            sorted_window = sorted(window)
            value_to_label = {value: index for index,
                              value in enumerate(sorted_window)}
            permuted_window = [value_to_label[value] for value in window]
            new_sequence.append(self.permutations.index(permuted_window))
        new_sequence.append(0)

        return new_sequence

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


class BCI_IV_2A_PERMUTED(PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, threshold=0.001, ):
        super().__init__(window_size, stride, BCI_IV_2A_DIR, train, threshold)


class BCI_IV_2B_PERMUTED(PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, threshold=0.001, ):
        super().__init__(window_size, stride, BCI_IV_2B_DIR, train, threshold)


class SIMPLE_PERMUTATION_DATASET(Dataset):
    def __init__(self, window_size, stride, data_dir, train=True, threshold=0.001, ):
        self.X, self.Y = fetch_data(data_dir)
        self.train = train
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold

        self.X = F.max_pool1d(self.X, 3, 3)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=43)
        if self.train:
            self.X_train = self._discretize(self.X_train)
            # self.X_train = self.X_train.to(torch.float32)
        else:
            self.X_val = self._discretize(self.X_val)
            # self.X_val = self.X_val.to(torch.float32)

    def get_X_shape(self):
        return self.X_train.shape

    def get_vocab_size(self):
        return len(self.permutations)

    def _discretize(self, X):
        # self.p_length = configs['BNCI2014_001']['window_size']
        values = [0, 1]
        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.window_size)]
        images, channels, timepoints = X.shape
        discretized_X = []
        for image in range(images):
            image_sequence = []
            for channel in range(channels):
                sequence = X[image, channel]
                discretized_sequence = self._permute(sequence)
                image_sequence.append(discretized_sequence)
            discretized_X.append(image_sequence)
        return torch.tensor(discretized_X, dtype=torch.long)

    def _permute(self, X):
        X = X.tolist()
        new_sequence = []
        X[0] = 0
        for i in range(1, len(X)):
            if X[i] > X[i-1]:
                X[i] = 1
            else:
                X[i] = 0
        # print(X)

        for i in range(0, len(X) - self.window_size, self.stride):
            window = X[i:(i+self.window_size)]
            new_sequence.append(self.permutations.index(window))
        new_sequence.append(0)

        return new_sequence

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


class SIMPLE_BCI_IV_2A_PERMUTED(SIMPLE_PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, threshold=0.001, ):
        super().__init__(window_size, stride, BCI_IV_2A_DIR, train, threshold)


class SIMPLE_BCI_IV_2B_PERMUTED(SIMPLE_PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, threshold=0.001, ):
        super().__init__(window_size, stride, BCI_IV_2B_DIR, train, threshold)
