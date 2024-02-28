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
from braindecode.models import ShallowFBCSPNet, EEGConformer
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
train_dataset = BNCI_LEFT_RIGHT(train=True)
val_dataset = BNCI_LEFT_RIGHT(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
# model = ShallowFBCSPNet(n_chans=22, n_classes=4,
#                         input_window_samples=1125, final_conv_length='auto')
model = TransWithEmbeddingV1(22, 2, 2, 64, 100)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=1125, final_conv_length='auto')
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%
x, y, = next(iter(train_dataloader))
x.shape, model(x).shape

# %%

train_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=True)
val_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
model = SimpleTransformerModelVanilla(input_dim=22, d_model=40)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=562, final_conv_length='auto')
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=200)
trainer.plot_train_val_scores()

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
