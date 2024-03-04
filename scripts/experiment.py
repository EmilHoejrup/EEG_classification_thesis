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
from torcheeg.models import SimpleViT, ATCNet, VanillaTransformer, EEGNet
from braindecode.models import ShallowFBCSPNet, EEGConformer
from torch.optim.lr_scheduler import LRScheduler
from new_model import *
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
# BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 64

has_gpu = torch.cuda.is_available()
device = 'cpu'
LEARNING_RATE = 0.001
EPOCHS = 10
# downsampling with moving average
# Hamming window?
# %%
train_dataset = BNCI_LEFT_RIGHT(train=True)
val_dataset = BNCI_LEFT_RIGHT(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True, )
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, )
# %%
# model = ShallowFBCSPNet(n_chans=22, n_classes=4,
#                         input_window_samples=1125, final_conv_length='auto')
# model = TransWithEmbeddingV1(22, 11, 8, 64, 64)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=1125, final_conv_length='auto')
model = Transformer(device='cpu', timepoints=561, seq_len=561)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%
x, y, = next(iter(train_dataloader))
x.shape
# %%
x.shape, model(x).shape

# %%

train_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=True)
val_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
# model = SimpleTransformerModelVanilla(
#     input_dim=22, d_model=40, num_layers=4, nhead=2)
model = Transformer(device='cpu', timepoints=562,
                    seq_len=562, n_classes=2, details=False)
# model = VanillaTransformer(num_electrodes=22)
# model = ATCNet(in_channels=22, num_classes=2,)
# model = EEGConformer(n_outputs=2, n_channels=64, )
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=562, final_conv_length='auto')
# model = TinyVGG(22,100,2)
loss_fun = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()

# %%
