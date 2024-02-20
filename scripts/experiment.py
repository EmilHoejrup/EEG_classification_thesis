# %%
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
import yaml
import numpy as np
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 64
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 30


# %%
# model = BasicMLP2(in_shape=INPUT_SIZE, out_shape=1, hidden_units=2000)
# running training loop with timer
# model = ImprovedMLP(in_shape=INPUT_SIZE, out_shape=1,
#                     hidden_units=200, num_layers=100)
# model = TinyVGG(input_shape=63, hidden_units=100, output_shape=1)
# model = NewBasic(n_channels=63, n_timepoints=300, hidden_units=30)
model = VanillaTransformerModel().to(device)


# %%
train_dataset = DiscretizedHimOrHer(train=True)
val_dataset = DiscretizedHimOrHer(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit()

# %%
x, y = next(iter(train_dataloader))
x[0], y[0]
