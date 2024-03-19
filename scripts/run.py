# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from datasets import *
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from torcheeg.models import SimpleViT, ATCNet, VanillaTransformer, EEGNet, ViT
from braindecode.models import ShallowFBCSPNet, EEGConformer
from torch.optim.lr_scheduler import LRScheduler
from new_model import *
from itertools import product

has_gpu = torch.cuda.is_available()
has_gpu = torch.cuda.is_available()
device = 'mps' if getattr(
    torch, 'torch.backends.mps.is_built()', False) else 'cuda' if has_gpu else 'cpu'


def run():
    with open(CONFIG_FILE, 'r') as file:
        configs = yaml.safe_load(file)
    dataset_name = configs['dataset']
    dataset_params = configs.get('datasets').get(dataset_name)
    param_combinations = product(*dataset_params.values())
    train_params = configs.get('train_params')
    if configs['dataset'] == 'BNCI_LEFT_RIGHT':
        for combination in param_combinations:
            train_dataset = BNCI_LEFT_RIGHT(*combination, train=True)
            val_dataset = BNCI_LEFT_RIGHT(*combination, train=False)
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
            _, _, timepoints = train_dataset.get_X_shape()
            train_models(train_dataloader, val_dataloader, timepoints)
            print(combination)


def train_models(train_dataloader, val_dataloader, timepoints):
    models = configs['models_to_train']
    for model in models:
        model_params = configs.get('models').get(model)
        print(model_params)
        param_combinations = product(*model_params.values())

        for combination in param_combinations:
            param_keys = model_params.keys()
            args = dict(zip(param_keys, combination))
            if model == 'EEGConformer':
                # model_params['input_window_samples'] = timepoints
                print(*args)
                model = EEGConformer(**args, input_window_samples=timepoints)

                print(f"model: {model}")

    # wandb.login()


# %%
run()

# %%
