# %%
"""Standardized benchmark taken from this tutorial: https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html#loading-preprocessing-defining-a-model-etc"""
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
import torch
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
import torch.nn.functional as F
import numpy as np
from braindecode.datasets import MOABBDataset

subject_ids = [1, 2, 3]
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)


low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False,
                 stim=False),  # Keep EEG sensors
    Preprocessor(
        lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
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

# %%

# check if GPU is available, if True chooses to use it
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]


class SimpleShallowNet(nn.Module):
    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75):
        super(SimpleShallowNet, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            in_channels, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels*maxpool_out, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = F.elu(self.spatio_temporal(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


models = []

shallowfbcspnet = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

simpleshallow = SimpleShallowNet(n_channels, n_classes, timepoints=1125)
# Display torchinfo table describing the model

# models.append(shallowfbcspnet)
models.append(simpleshallow)

final_metrics = []


def run(model):
    print(model)
    # Send model to GPU
    if cuda:
        model.cuda()

    splitted = windows_dataset.split("session")
    train_set = splitted['0train']  # Session train
    test_set = splitted['1test']  # Session evaluation

    # lr = 0.0625 * 0.01
    lr = 0.0001
    weight_decay = 0
    batch_size = 64
    n_epochs = 2700

    # Define a method for training one epoch

    def train_one_epoch(
            dataloader: DataLoader, model: Module, loss_fn, optimizer,
            scheduler: LRScheduler, epoch: int, device, print_batch_stats=True
    ):
        model.train()  # Set the model to training mode
        train_loss, correct = 0, 0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                            disable=not print_batch_stats)

        for batch_idx, (X, y, _) in progress_bar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()  # update the model weights
            optimizer.zero_grad()

            train_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

            if print_batch_stats:
                progress_bar.set_description(
                    f"Epoch {epoch}/{n_epochs}, "
                    f"Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        # Update the learning rate
        scheduler.step()

        correct /= len(dataloader.dataset)
        return train_loss / len(dataloader), correct

    @torch.no_grad()
    def test_model(
        dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True
    ):
        size = len(dataloader.dataset)
        n_batches = len(dataloader)
        model.eval()  # Switch to evaluation mode
        test_loss, correct = 0, 0

        if print_batch_stats:
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            progress_bar = enumerate(dataloader)

        for batch_idx, (X, y, _) in progress_bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            batch_loss = loss_fn(pred, y).item()

            test_loss += batch_loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if print_batch_stats:
                progress_bar.set_description(
                    f"Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {batch_loss:.6f}"
                )

        test_loss /= n_batches
        correct /= size

        print(
            f"Test Accuracy: {100 * correct:.1f}%, Test Loss: {test_loss:.6f}\n"
        )
        return test_loss, correct

    # Define the optimization
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=n_epochs - 1)
    # Define the loss function
    # We used the NNLoss function, which expects log probabilities as input
    # (which is the case for our model output)
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    # train_set and test_set are instances of torch Datasets, and can seamlessly be
    # wrapped in data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}: ", end="")

        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device,
        )

        test_loss, test_accuracy = test_model(test_loader, model, loss_fn)

        print(
            f"Train Accuracy: {100 * train_accuracy:.2f}%, "
            f"Average Train Loss: {train_loss:.6f}, "
            f"Test Accuracy: {100 * test_accuracy:.1f}%, "
            f"Average Test Loss: {test_loss:.6f}\n"
        )
    model_metrics = {
        "model": model.__class__.__name__,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    final_metrics.append(model_metrics)


def __main__():
    for model in models:
        run(model)
    print(final_metrics)


if __name__ == "__main__":
    __main__()

# %%
