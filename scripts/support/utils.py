"""Preprocessing steps taken from tutorial: https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html#loading-preprocessing-defining-a-model-etc
"""
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from braindecode.preprocessing import create_windows_from_events
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
BNCI_4_CLASS_DIR = DATA_DIR / 'BNCI_4_CLASS'


def fetch_preprocess_and_save_BNCI2014_001_4_classes():
    dataset = MOABBDataset(dataset_name='BNCI2014_001', subject_ids=[3])
    dataset = preprocess_dataset(dataset)
    dataset = create_windows_dataset(dataset)
    save_BNCI_4_class(dataset)


def preprocess_dataset(dataset):
    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    transforms = [
        Preprocessor("pick_types", eeg=True, meg=False,
                     stim=False),  # Keep EEG sensors
        Preprocessor(
            lambda data, factor: np.multiply(
                data, factor),  # Convert from V to uV
            factor=1e6,
        ),
        Preprocessor("filter", l_freq=low_cut_hz,
                     h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(
            exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]

    # Transform the data
    preprocess(dataset, transforms, n_jobs=-1)
    return dataset


def create_windows_dataset(dataset):
    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )
    return windows_dataset


def save_BNCI_4_class(dataset):
    BNCI_4_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save(
        path=BNCI_4_CLASS_DIR,
        overwrite=True
    )


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
