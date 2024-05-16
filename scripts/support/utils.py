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
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
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


has_gpu = torch.cuda.is_available()
device = 'mps' if getattr(
    torch, 'torch.backends.mps.is_built()', False) else 'cuda' if has_gpu else 'cpu'
# %%


def fetch_BNCI2014_001():
    dataset = BNCI2014_001()
    subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset.subject_list = subject_list
    paradigm = LeftRightImagery()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=subject_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train = X[meta['session'] == '0train']
    y_train = y[meta['session'] == '0train']
    X_test = X[meta['session'] == '1test']
    y_test = y[meta['session'] == '1test']
    # Save data
    BCI_2B_TRAIN.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_2B_TRAIN, 'wb') as f:
        np.save(f, X_train)
    with open(Y_BCI_2B_TRAIN, 'wb') as f:
        pickle.dump(y_train, f)
    BCI_2B_TEST.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_2B_TEST, 'wb') as f:
        np.save(f, X_test)
    with open(Y_BCI_2B_TEST, 'wb') as f:
        pickle.dump(y_test, f)


def fetch_BCI_IV_2A():
    dataset = BNCI2014_001()
    subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset.subject_list = subject_list
    paradigm = MotorImagery()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=subject_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train = X[meta['session'] == '0train']
    y_train = y[meta['session'] == '0train']
    X_test = X[meta['session'] == '1test']
    y_test = y[meta['session'] == '1test']
    # Save data
    BCI_2A_TRAIN.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_2A_TRAIN, 'wb') as f:
        np.save(f, X_train)
    with open(Y_BCI_2A_TRAIN, 'wb') as f:
        pickle.dump(y_train, f)
    BCI_2A_TEST.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BCI_2A_TEST, 'wb') as f:
        np.save(f, X_test)
    with open(Y_BCI_2A_TEST, 'wb') as f:
        pickle.dump(y_test, f)

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


def test_metrics(model, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    test_acc = 0
    model.eval()
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = criterion(y_logits, y)
        _, predicted = torch.max(y_logits, 1)
        test_acc += (predicted == y).sum().item()
    test_acc /= len(test_dataloader.dataset)
    if len(y.unique()) > 2:
        weighted_kappa = cohen_kappa_score(
            y.cpu().numpy(), predicted.cpu().numpy(), weights='quadratic')
        weighted_precision = precision_score(
            y.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        weighted_recall = recall_score(
            y.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        return test_acc, weighted_kappa, weighted_precision, weighted_recall
    else:
        weighted_kappa = cohen_kappa_score(
            y.cpu().numpy(), predicted.cpu().numpy())
        weighted_precision = precision_score(
            y.cpu().numpy(), predicted.cpu().numpy())
        weighted_recall = recall_score(
            y.cpu().numpy(), predicted.cpu().numpy())
        return test_acc, weighted_kappa, weighted_precision, weighted_recall
