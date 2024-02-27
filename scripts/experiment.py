# %%
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import math
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
from support.utils import *
import yaml
import numpy as np
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from braindecode.models import ShallowFBCSPNet
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 64
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 50
# downsampling with moving average
# Hamming window?
# %%
# Define a method for training one epoch


bnci_4_class = BNCI_4_CLASS()
train_dataset, val_dataset = bnci_4_class.get_train_and_test_data()
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# model = ShallowFBCSPNet(n_chans=22, n_classes=4,
#                         input_window_samples=1125, final_conv_length='auto')
model = SimpleTransformerModelVanilla(input_dim=22)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%
x, y, _ = next(iter(train_dataloader))
x.shape, model(x).shape

# %%


class DiscretizeTransformer:
    def __init__(self, window_size, stride):
        values = [1, 0]
        self.window_size = window_size
        self.stride = stride
        self.permutations = [list(perm)
                             for perm in product(values, repeat=window_size)]

    def __call__(self, channels):
        transformed_channels = []
        for sequence in channels:
            new_sequence = self._permute(sequence)
            transformed_channels.append(new_sequence)

        return transformed_channels

    def _permute(self, sequence):
        new_sequence = []
        sequence[0] = 0
        for i in range(1, len(sequence)):
            if sequence[i] > sequence[i-1]:
                sequence[i] = 1
            else:
                sequence[i] = 0
        # loop through the with the specified window size and stride as step
        for i in range(0, len(sequence) - self.window_size, self.stride):
            window = list(sequence[i:(i+self.window_size)])
            new_sequence.append(self.permutations.index(window))
        return new_sequence


custom_transform = DiscretizeTransformer(6, 2)


def custom_collate(batch):
    x, y, _ = zip(*batch)
    x = list(x)
    # print(x)
    # print(y)
    # print(_)

    for i in range(len(batch)):
        x[i] = (custom_transform(x[i]))
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y)
    _ = torch.tensor(_)
    return x, y, _
    # print(batch)
    # transformed_batch = [
    #     custom_transform(sample) for sample in batch]

    # print(type(transformed_batch))
    # # print(len(transformed_batch))
    # # # transformed_batch = [torch.tensor(sample) for sample in transformed_batch]
    # # t = torch.tensor(transformed_batch[0])
    # # print(t)
    # return batch


# %%
x, y, _ = next(iter(train_dataloader))


# %%
bnci_4_class = BNCI_4_CLASS()
train_dataset, val_dataset = bnci_4_class.get_train_and_test_data()
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)

# model = ShallowFBCSPNet(n_chans=22, n_classes=4,
#                         input_window_samples=1125, final_conv_length='auto')
model = TransWithEmbeddingV1(22, 2, 2, 64, 60)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%
x, y, _ = next(iter(train_dataloader))
x.shape, model(x).shape
