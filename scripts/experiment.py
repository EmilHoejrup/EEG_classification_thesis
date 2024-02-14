# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import HimOrHer
import yaml
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader

with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']

train_dataset = HimOrHer(train=True)
val_dataset = HimOrHer(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE,  shuffle=True)

for batch_x, batch_y in train_dataloader:
    print(batch_y)

# %%
