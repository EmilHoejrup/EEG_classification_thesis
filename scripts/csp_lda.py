# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
import mne
from mne.decoding import CSP
from support.constants import DATA_DIR, BNCI_CHANNELS, BNCI_SFREQ
from datasets import read_X_and_y
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

BCI_2A = DATA_DIR / 'BCI_IV_2A'
BCI_2A_TRAIN = BCI_2A / 'train'
BCI_2A_TRAIN_X = BCI_2A_TRAIN / 'X.npy.gz'
BCI_2A_TRAIN_Y = BCI_2A_TRAIN / 'Y.p'
BCI_2A_TEST = BCI_2A / 'test'
BCI_2A_TEST_X = BCI_2A_TEST / 'X.npy.gz'
BCI_2A_TEST_Y = BCI_2A_TEST / 'Y.p'

BCI_2B = DATA_DIR / 'BCI_IV_2B'
BCI_2B_TRAIN = BCI_2B / 'train'
BCI_2B_TRAIN_X = BCI_2B_TRAIN / 'X.npy.gz'
BCI_2B_TRAIN_Y = BCI_2B_TRAIN / 'Y.p'
BCI_2B_TEST = BCI_2B / 'test'
BCI_2B_TEST_X = BCI_2B_TEST / 'X.npy.gz'
BCI_2B_TEST_Y = BCI_2B_TEST / 'Y.p'


def x_to_epochs(X):
    images, channels, timepoints = X.shape

    channel_types = ['eeg'] * channels
    info = mne.create_info(BNCI_CHANNELS, BNCI_SFREQ, ch_types=channel_types)
    # Creating events to be simply the beginning and end of timeseries
    events = [[i, 0, 1] for i in range(len(X))]
    epochs = mne.EpochsArray(data=X.numpy(), info=info, events=events, tmin=0)
    # epochs.drop_channels('P4')
    return epochs


# The following part is taken from this MNE tutorial:https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
def train(epochs_array_train, train_labels, epochs_array_test, test_labels):
    epochs_train = epochs_array_train.copy()
    epochs_test = epochs_array_test.copy()
    epochs_data_train = epochs_train.get_data(copy=False)
    epochs_data_test = epochs_test.get_data(copy=False)

    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    clf.fit(epochs_data_train, train_labels)
    clf_score = clf.score(epochs_data_test, test_labels)
    # Printing the results
    class_balance = np.mean(train_labels == train_labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    num_classes = len(np.unique(train_labels))
    chance_level = 1 / num_classes
    return clf_score, chance_level


def train_2a():
    X_train, Y_train = read_X_and_y(BCI_2A_TRAIN_X, BCI_2A_TRAIN_Y)
    X_test, Y_test = read_X_and_y(BCI_2A_TEST_X, BCI_2A_TEST_Y)

    epochs_array_train = x_to_epochs(X_train)
    epochs_array_test = x_to_epochs(X_test)

    clf_score, chance_level = train(
        epochs_array_train, Y_train, epochs_array_test, Y_test)
    return clf_score, chance_level


def train_2b():
    X_train, Y_train = read_X_and_y(BCI_2B_TRAIN_X, BCI_2B_TRAIN_Y)
    X_test, Y_test = read_X_and_y(BCI_2B_TEST_X, BCI_2B_TEST_Y)

    epochs_array_train = x_to_epochs(X_train)
    epochs_array_test = x_to_epochs(X_test)

    clf_score, chance_level = train(
        epochs_array_train, Y_train, epochs_array_test, Y_test)
    return clf_score, chance_level


if __name__ == '__main__':
    print("Training BCI IV 2a")
    clf_2a, chance_level_2a = train_2a()
    print("Training BCI IV 2b")
    clf_2b, chance_level_2b = train_2b()
    print(f"BCI IV 2a accuracy: {clf_2a}, chance level: {chance_level_2a}")
    print(f"BCI IV 2b accuracy: {clf_2b}, chance level: {chance_level_2b}")
