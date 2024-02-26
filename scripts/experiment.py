# %%
import math
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
import yaml
import numpy as np
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from transforms import *
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 32
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 300
# downsampling with moving average
# Hamming window?
# %%
train_dataset = BNCI2014_001_DISCRETIZED(train=True)
val_dataset = BNCI2014_001_DISCRETIZED(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
x, y = next(iter(train_dataloader))
x.shape
# %%
trans_with_emb_params = configs['BNCTransWithEmbeddingV1']['model_params']
# model = TransWithEmbeddingV1(**trans_with_emb_params)
model = TransWithEmbeddingV1(
    d_model=22, nhead=2, num_layers=4, vocab_size=1100, emb_dim=64)

loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()

# %%

train_dataset = BNCI2014_001(train=True)
val_dataset = BNCI2014_001(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
x, y = next(iter(train_dataloader))
x.shape
# %%
trans_with_emb_params = configs['TransWithEmbeddingV1']['model_params']
model = SimpleTransformerModelVanilla(input_dim=22, d_model=300)
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()
# %%

x, y = next(iter(train_dataloader))
# %%
