import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as p
import gzip
from pathlib import Path
from itertools import product
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
X_BNCI2015_001 = DATA_DIR / 'BNCI2014_001' / 'X.npy.gz'
Y_BNCI2015_001 = DATA_DIR / 'BNCI2014_001' / 'Y.p'
BNCI_DISCRETIZED_DIR = DATA_DIR / 'BNCI_DISCRETIZED'
BNCI_DECONSTRUCTED = DATA_DIR / 'BNCI_deconstructed'


def read_X_and_y(x_file, y_file):
    with gzip.GzipFile(x_file, "r") as f:
        X = torch.from_numpy(np.load(file=f))

    with open(y_file, 'rb') as f:
        Y = p.load(f)
    return X, Y


class BNCI_LEFT_RIGHT_CONTINUOUS(Dataset):
    def __init__(self, train):
        if not BNCI_DECONSTRUCTED.exists():
            bnci_4 = BNCI_4_CLASS()
            deconstruct_moabb_dataset(bnci_4, BNCI_DECONSTRUCTED)
        self.train = train

        samples_0 = torch.load(BNCI_DECONSTRUCTED / '0_samples.pth')
        samples_1 = torch.load(BNCI_DECONSTRUCTED / '1_samples.pth')
        labels_0 = torch.load(BNCI_DECONSTRUCTED / '0_labels.pth')
        labels_1 = torch.load(BNCI_DECONSTRUCTED / '1_labels.pth')
        self.X = torch.cat((samples_0, samples_1), dim=0)
        self.Y = torch.cat((labels_0, labels_1), dim=0)
        self.X = self.X.to(torch.float32)
        self.X = F.avg_pool1d(self.X, 3, 2)

        # self.Y = self.Y.astype(float)
        # self.Y = self.Y.to(torch.float32)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=43)

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


class BNCI_LEFT_RIGHT(Dataset):
    def __init__(self, train):
        if not BNCI_DECONSTRUCTED.exists():
            bnci_4 = BNCI_4_CLASS()
            deconstruct_moabb_dataset(bnci_4, BNCI_DECONSTRUCTED)
        self.train = train

        samples_0 = torch.load(BNCI_DECONSTRUCTED / '0_samples.pth')
        samples_1 = torch.load(BNCI_DECONSTRUCTED / '1_samples.pth')
        labels_0 = torch.load(BNCI_DECONSTRUCTED / '0_labels.pth')
        labels_1 = torch.load(BNCI_DECONSTRUCTED / '1_labels.pth')
        self.X = torch.cat((samples_0, samples_1), dim=0)
        self.Y = torch.cat((labels_0, labels_1), dim=0)

        # self.Y = self.Y.astype(float)
        # self.Y = self.Y.to(torch.float32)
        self.X = self._discretize(self.X)
        self.X = self.X.to(torch.float32)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=43)

    def _discretize(self, X):
        self.p_length = configs['BNCI2014_001']['window_size']
        values = [0, 1]
        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.p_length)]
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

        for i in range(0, len(X) - self.p_length, 2):
            window = X[i:(i+self.p_length)]
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


class BNCI_4_CLASS(Dataset):
    def __init__(self):
        if not BNCI_4_CLASS_DIR.exists():
            fetch_preprocess_and_save_BNCI2014_001_4_classes()
        self.dataset = load_concat_dataset(
            BNCI_4_CLASS_DIR, preload=False, target_name=None)

    def get_train_and_test_data(self):
        splitted = self.dataset.split('session')
        return splitted['0train'], splitted['1test']

    def __getitem__(self, index):
        return self.dataset[index]


def deconstruct_moabb_dataset(dataset, path):
    class_samples = {}
    for sample, label, _ in dataset:
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append((sample, label))
    path.mkdir(parents=True, exist_ok=True)
    for label, sample in class_samples.items():
        samples, labels = zip(*sample)
        samples = torch.tensor(samples)
        labels = torch.tensor(labels)
        sample_path = path / f"{label}_samples.pth"
        torch.save(samples, sample_path)
        label_path = path / f"{label}_labels.pth"
        torch.save(labels, label_path)

##################### OLD DATASETS #####################


class BNCI2014_001_DISCRETIZED(Dataset):
    def __init__(self, train=True, val_split=0.2):
        self.train = train
        if not X_BNCI2015_001.exists():
            fetch_BNCI2014_001()
        self.X, self.Y = read_X_and_y(X_BNCI2015_001, Y_BNCI2015_001)
        # self.X = F.avg_pool1d(self.X, 3, 2)
        self.X = self.X[:, :, :300]
        self.Y = self.Y.astype(float)
        # self.X, self.Y = self.X[:300], self.Y[:300]
        self.X = self._discretize(self.X)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=val_split, random_state=43)

    def _discretize(self, X):
        self.p_length = configs['BNCI2014_001']['window_size']
        values = [0, 1]
        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.p_length)]
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

        for i in range(0, len(X) - self.p_length, 2):
            window = X[i:(i+self.p_length)]
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


class DiscretizedHimOrHer(Dataset):
    def __init__(self, train=True, val_split=0.2):
        self.train = train
        self.X, self.Y = read_X_and_y(X_HIM_OR_HER, Y_HIM_OR_HER)
        # self.X, self.Y = self.X[:10], self.Y[:10]
        self.X = self._discretize(self.X)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=val_split, random_state=43)

    def _discretize(self, X):
        self.p_length = configs['him_or_her']['window_size']
        values = [0, 1]
        self.permutations = [list(perm)
                             for perm in product(values, repeat=self.p_length)]
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

        for i in range(0, len(X) - self.p_length, 2):
            window = X[i:(i+self.p_length)]
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


class HimOrHer(Dataset):
    def __init__(self, train=True, val_split=0.2):
        self.train = train
        self.X, self.Y = read_X_and_y(X_HIM_OR_HER, Y_HIM_OR_HER)

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


class BNCI2014_001(Dataset):
    def __init__(self, train=True, val_split=0.2):
        self.train = train
        if not X_BNCI2015_001.exists():
            fetch_BNCI2014_001()
        self.X, self.Y = read_X_and_y(X_BNCI2015_001, Y_BNCI2015_001)
        # self.X, self.Y = self.X[:300], self.Y[:300]
        self.X = self.X[:, :, 500:850]
        self.X = F.avg_pool1d(self.X, 3, 2)
        self.X = self.X.type(torch.float32)
        self.Y = self.Y.astype(float)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=val_split, random_state=43)
        # self.X_train = self.z_score_normalization(self.X_train)
        # self.X_val = self.z_score_normalization(self.X_val)

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

    def z_score_normalization(self, tensor, axis=2):
        mean = torch.mean(tensor, dim=axis, keepdim=True)
        std = torch.std(tensor, dim=axis, keepdim=True)
        # Adding small number to avoid division by zero
        tensor = (tensor - mean) / (std + 1e-8)
        return tensor
