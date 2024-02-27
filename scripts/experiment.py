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
BATCH_SIZE = 32
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

model = ShallowFBCSPNet(n_chans=22, n_classes=4,
                        input_window_samples=1125, final_conv_length='auto')

loss_fun = nn.NLLLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
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
    d_model=22, nhead=2, num_layers=4, vocab_size=64, emb_dim=64)
# model = VanillaTransformerModel(input_dim=497, d_model=64)

loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
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
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%

x, y = next(iter(train_dataloader))
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
# model = TinyVGG(22, 200, 1)
model = EEGNet(2)
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=EPOCHS)
trainer.plot_train_val_scores()
# %%


# %%
