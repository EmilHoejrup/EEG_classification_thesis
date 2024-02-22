import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from support.constants import DATA_DIR
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder
BNCI2014_001_DIR = DATA_DIR / 'BNCI2014_001'
X_BNCI2015_001 = BNCI2014_001_DIR / 'X.npy.gz'
Y_BNCI2015_001 = BNCI2014_001_DIR / 'Y.p'


def fetch_BNCI2014_001():
    dataset = BNCI2014_001()
    dataset.subject_list = [1, 2, 3]
    paradigm = LeftRightImagery()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # Save data
    BNCI2014_001_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(X_BNCI2015_001, 'wb') as f:
        np.save(f, X)
    with open(Y_BNCI2015_001, 'wb') as f:
        pickle.dump(y, f)


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