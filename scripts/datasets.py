import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as p
import gzip
from itertools import product
from support.constants import *
import yaml
from support.constants import CONFIG_FILE
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
from support.utils import *

BCI_IV_2B_DIR = DATA_DIR / 'BCI_IV_2B'
BCI_2B_TRAIN = BCI_IV_2B_DIR / 'train'
X_BCI_2B_TRAIN = BCI_2B_TRAIN / 'X.npy.gz'
Y_BCI_2B_TRAIN = BCI_2B_TRAIN / 'Y.p'
BCI_2B_TEST = BCI_IV_2B_DIR / 'test'
X_BCI_2B_TEST = BCI_2B_TEST / 'X.npy.gz'
Y_BCI_2B_TEST = BCI_2B_TEST / 'Y.p'

BCI_IV_2A_DIR = DATA_DIR / 'BCI_IV_2A'
BCI_2A_TRAIN = BCI_IV_2A_DIR / 'train'
X_BCI_2A_TRAIN = BCI_2A_TRAIN / 'X.npy.gz'
Y_BCI_2A_TRAIN = BCI_2A_TRAIN / 'Y.p'
BCI_2A_TEST = BCI_IV_2A_DIR / 'test'
X_BCI_2A_TEST = BCI_2A_TEST / 'X.npy.gz'
Y_BCI_2A_TEST = BCI_2A_TEST / 'Y.p'


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
        X_train, Y_train = read_X_and_y(X_BCI_2A_TRAIN, Y_BCI_2A_TRAIN)
        X_test, Y_test = read_X_and_y(X_BCI_2A_TEST, Y_BCI_2A_TEST)

    elif data_dir == BCI_IV_2B_DIR:
        if not BCI_IV_2B_DIR.exists():
            fetch_BNCI2014_001()
        X_train, Y_train = read_X_and_y(X_BCI_2B_TRAIN, Y_BCI_2B_TRAIN)
        X_test, Y_test = read_X_and_y(X_BCI_2B_TEST, Y_BCI_2B_TEST)
    return X_train, Y_train, X_test, Y_test


class CONTINUOUS_DATASET(Dataset):
    """ The default dataset class without permutation pattern preprocessing"""

    def __init__(self, data_dir, train=True, val=False, test=False):
        self.X, self.Y, self.X_test, self.Y_test = fetch_data(
            data_dir)

        self.X = self.X.to(torch.float32)
        self.X_test = self.X_test.to(torch.float32)
        target_mean = torch.mean(self.X, dim=(0, 2), keepdim=True)
        target_std = torch.std(self.X, dim=(0, 2), keepdim=True)
        self.X = (self.X - target_mean) / target_std
        self.X_test = (self.X_test - target_mean) / target_std

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)
        self.train = train
        self.val = val
        self.test = test

    def get_X_shape(self):
        if self.train:
            return self.X_train.shape
        elif self.val:
            return self.X_val.shape
        else:
            return self.X_test.shape

    def __len__(self):
        if self.train:
            return len(self.X_train)
        elif self.val:
            return len(self.X_val)
        else:
            return len(self.X_test)

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.Y_train[index]
        elif self.val:
            return self.X_val[index], self.Y_val[index]
        else:
            return self.X_test[index], self.Y_test[index]


class BCI_2A_CONTINUOUS(CONTINUOUS_DATASET):
    def __init__(self, train=True, val=False, test=False):
        super().__init__(BCI_IV_2A_DIR, train, val, test)


class BCI_2B_CONTINUOUS(CONTINUOUS_DATASET):
    def __init__(self, train=True, val=False, test=False):
        super().__init__(BCI_IV_2B_DIR, train, val, test)


class PERMUTATION_DATASET(Dataset):
    """Dataset class implementing the permutation pattern preprocessing"""

    def __init__(self, window_size, stride, data_dir, train=True, val=False, test=False):
        self.X, self.Y, self.X_test, self.Y_test = fetch_data(
            data_dir)

        self.train = train
        self.val = val
        self.test = test

        self.window_size = window_size
        self.stride = stride

        self.X = F.max_pool1d(self.X, 3, 3)
        self.X = self._discretize(self.X)
        self.X_test = F.max_pool1d(self.X_test, 3, 3)
        self.X_test = self._discretize(self.X_test)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)

    def get_X_shape(self):
        if self.train:
            return self.X_train.shape
        elif self.val:
            return self.X_val.shape
        else:
            return self.X_test.shape

    def get_vocab_size(self):
        return len(self.permutations)

    def _discretize(self, X):
        values = list(range(self.window_size))

        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.window_size)]
        trials, channels, timepoints = X.shape
        discretized_X = []
        for trial in range(trials):
            trial_sequence = []
            for channel in range(channels):
                sequence = X[trial, channel]
                discretized_sequence = self._permute(sequence)
                trial_sequence.append(discretized_sequence)
            discretized_X.append(trial_sequence)
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
        elif self.val:
            return len(self.X_val)
        else:
            return len(self.X_test)

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.Y_train[index]
        elif self.val:
            return self.X_val[index], self.Y_val[index]
        else:
            return self.X_test[index], self.Y_test[index]


class BCI_2A_PERMUTED(PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, val=False, test=False):
        super().__init__(window_size, stride, BCI_IV_2A_DIR, train, val, test)


class BCI_2B_PERMUTED(PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, val=False, test=False):
        super().__init__(window_size, stride, BCI_IV_2B_DIR, train, val, test)


class SIMPLE_PERMUTATION_DATASET(Dataset):
    """ Dataset class implementing the simplified version of permutation pattern preprocessing"""

    def __init__(self, window_size, stride, data_dir, train=True, val=False, test=False):
        self.X, self.Y, self.X_test, self.Y_test = fetch_data(
            data_dir)

        self.train = train
        self.val = val
        self.test = test
        self.window_size = window_size
        self.stride = stride

        self.X = F.max_pool1d(self.X, 3, 3)
        self.X = self._discretize(self.X)
        self.X_test = F.max_pool1d(self.X_test, 3, 3)
        self.X_test = self._discretize(self.X_test)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)

    def get_X_shape(self):
        if self.train:
            return self.X_train.shape
        elif self.val:
            return self.X_val.shape
        else:
            return self.X_test.shape

    def get_vocab_size(self):
        return len(self.permutations)

    def _discretize(self, X):
        values = [0, 1]
        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.window_size)]
        trials, channels, timepoints = X.shape
        discretized_X = []
        for trial in range(trials):
            trial_sequence = []
            for channel in range(channels):
                sequence = X[trial, channel]
                discretized_sequence = self._permute(sequence)
                trial_sequence.append(discretized_sequence)
            discretized_X.append(trial_sequence)
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
        elif self.val:
            return len(self.X_val)
        else:
            return len(self.X_test)

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.Y_train[index]
        elif self.val:
            return self.X_val[index], self.Y_val[index]
        else:
            return self.X_test[index], self.Y_test[index]


class SIMPLE_BCI_2A_PERMUTED(SIMPLE_PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, val=False, test=False):
        super().__init__(window_size, stride, BCI_IV_2A_DIR, train, val, test)


class SIMPLE_BCI_2B_PERMUTED(SIMPLE_PERMUTATION_DATASET):
    def __init__(self, window_size, stride, train=True, val=False, test=False):
        super().__init__(window_size, stride, BCI_IV_2B_DIR, train, val, test)
