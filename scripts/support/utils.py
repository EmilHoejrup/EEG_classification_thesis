"""Preprocessing steps taken from tutorial: https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html#loading-preprocessing-defining-a-model-etc
"""
# %%

from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery, MotorImagery
from support.constants import DATA_DIR
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder
# BNCI2014_001_DIR = DATA_DIR / 'BNCI2014_001'
# X_BNCI2014_001 = BNCI2014_001_DIR / 'X.npy.gz'
# Y_BNCI2014_001 = BNCI2014_001_DIR / 'Y.p'
# BNCI2014_002_DIR = DATA_DIR / 'BNCI2014_002'
# X_BNCI2014_002 = BNCI2014_002_DIR / 'X.npy.gz'
# Y_BNCI2014_002 = BNCI2014_002_DIR / 'Y.p'

BCI_IV_2B_DIR = DATA_DIR / 'BCI_IV_2B'
X_BCI_IV_2B = BCI_IV_2B_DIR / 'X.npy.gz'
Y_BCI_IV_2B = BCI_IV_2B_DIR / 'Y.p'
BCI_IV_2A_DIR = DATA_DIR / 'BCI_IV_2A'
X_BCI_IV_2A = BCI_IV_2A_DIR / 'X.npy.gz'
Y_BCI_IV_2A = BCI_IV_2A_DIR / 'Y.p'


def fetch_BNCI2014_001():
    dataset = BNCI2014_001()
    dataset.subject_list = [1, 2, 3]
    paradigm = LeftRightImagery()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # Save data
    BCI_IV_2B_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_IV_2B, 'wb') as f:
        np.save(f, X)
    with open(Y_BCI_IV_2B, 'wb') as f:
        pickle.dump(y, f)


def fetch_BCI_IV_2A():
    dataset = BNCI2014_001()
    dataset.subject_list = [1, 2, 3]
    paradigm = MotorImagery()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # Save data
    BCI_IV_2A_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_IV_2A, 'wb') as f:
        np.save(f, X)
    with open(Y_BCI_IV_2A, 'wb') as f:
        pickle.dump(y, f)

# %%


def plot_train_val_scores(train_loss, train_acc, val_loss, val_acc):
    epochs = range(len(train_loss))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, val_loss, label='validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
