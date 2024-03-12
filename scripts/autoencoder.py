# %%
import itertools
from pytorch_lightning import Trainer
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
LEARNING_RATE = 0.0001
EPOCHS = 3

# %%


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.input_dim)
        )

    def forward(self, x):
        batch, channels, timepoints = x.shape
        x = x.view(batch, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.shape)
        return x
