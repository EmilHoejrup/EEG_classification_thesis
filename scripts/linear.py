# %%
import gzip
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
import mne
from mne.decoding import CSP
import torch
import pickle as p
import yaml
from support.constants import DATA_DIR, BNCI_CHANNELS, BNCI_SFREQ
from datasets import load_bnci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import moabb
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
import seaborn as sns
import matplotlib as plt
# mne.set_log_level(verbose='WARNING')
# HIM_OR_HER = DATA_DIR / 'Processed' / 'him-or-her'
# X_FILE = HIM_OR_HER / 'X.npy.gz'
# Y_FILE = HIM_OR_HER / 'Y.p'
# with open(HIM_OR_HER / 'prep_configs.yaml', 'r') as f:
#     prep_configs = yaml.safe_load(f)


# def read_X_and_y(x_file, y_file):
#     with gzip.GzipFile(x_file, "r") as f:
#         X = torch.from_numpy(np.load(file=f))

#     with open(y_file, 'rb') as f:
#         Y = p.load(f)
#     return X, Y
from moabb.datasets.base import BaseDataset
import numpy as np


def x_to_epochs(X):
    images, channels, timepoints = X.shape

    channel_types = ['eeg'] * channels
    info = mne.create_info(BNCI_CHANNELS, BNCI_SFREQ, ch_types=channel_types)
    # Creating events to be simply the beginning and end of timeseries
    events = [[i, 0, 1] for i in range(len(X))]
    epochs = mne.EpochsArray(data=X.numpy(), info=info, events=events, tmin=0)
    # epochs.drop_channels('P4')
    return epochs


# The following part taken from this MNE tutorial:https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
def train(epochs_array, labels):
    epochs_train = epochs_array.copy()
    scores = []
    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=80, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train,
                             labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(
        f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")


if __name__ == '__main__':
    X, Y = load_bnci()
    X.shape
# %%
    epochs_array = x_to_epochs(X)
    labels = Y.numpy()
    train(epochs_array, labels)
