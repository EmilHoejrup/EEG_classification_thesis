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
from itertools import product
from trainer import MultiLabelClassifierTrainer
from EEGTransformer import EEGTransformer, EEGTransformerEmb
from ConformerCopy import ConformerCopy
from GraphFormer import GraphFormer

has_gpu = torch.cuda.is_available()
has_gpu = torch.cuda.is_available()
device = 'mps' if getattr(
    torch, 'torch.backends.mps.is_built()', False) else 'cuda' if has_gpu else 'cpu'


def run():
    wandb.login()
    with open(CONFIG_FILE, 'r') as file:
        configs = yaml.safe_load(file)
    dataset_name = configs['dataset']
    dataset_params = configs.get('datasets').get(dataset_name)
    train_params = configs.get('train_params')
    if configs['dataset'] == 'BNCI_LEFT_RIGHT' or configs['dataset'] == 'BNCI_LEFT_RIGHT_NEW_PE':
        param_combinations = product(*dataset_params.values())
        for combination in param_combinations:
            train_dataset = BNCI_LEFT_RIGHT(*combination, train=True)
            val_dataset = BNCI_LEFT_RIGHT(*combination, train=False)
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
            _, _, timepoints = train_dataset.get_X_shape()
            vocab_size = train_dataset.get_vocab_size()
            train_models(train_dataloader, val_dataloader,
                         timepoints, combination, vocab_size=vocab_size)

    elif configs['dataset'] == 'BNCI_LEFT_RIGHT_CONTINUOUS':
        train_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=True)
        val_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=False)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
        _, _, timepoints = train_dataset.get_X_shape()
        train_models(train_dataloader, val_dataloader,
                     timepoints)

    elif configs['dataset'] == 'BNCI_LEFT_RIGHT_COMPRESSED':
        param_combinations = product(*dataset_params.values())
        for combination in param_combinations:
            train_dataset = BNCI_LEFT_RIGHT_COMPRESSED(
                *combination, train=True)
            val_dataset = BNCI_LEFT_RIGHT_COMPRESSED(*combination, train=False)
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
            _, timepoints = train_dataset.get_X_shape()
            vacab_size = train_dataset.get_vocab_size()
            train_models(train_dataloader, val_dataloader,
                         timepoints, combination, vocab_size=vacab_size)


def train_models(train_dataloader, val_dataloader, timepoints, dataset_combination=None, configs=configs, vocab_size=None):
    models = configs['models_to_train']
    for model_type in models:
        model_params = configs.get('models').get(model_type)
        # print(model_params)
        param_combinations = product(*model_params.values())

        for combination in param_combinations:
            param_keys = model_params.keys()
            args = dict(zip(param_keys, combination))
            if model_type == 'EEGConformer':
                model = EEGConformer(**args, n_times=timepoints)
            elif model_type == 'VanillaTransformer':
                model = VanillaTransformer(
                    chunk_size=timepoints, t_patch_size=timepoints//5, **args)
            elif model_type == 'EEGTransformer':
                model = EEGTransformer(
                    **args, seq_len=timepoints, vocab_size=vocab_size)
            elif model_type == 'EEGTransformerEmb':
                model = EEGTransformerEmb(
                    **args, seq_len=timepoints, vocab_size=vocab_size)
            elif model_type == 'ConformerCopy':
                model = ConformerCopy(
                    **args, vocab_size=vocab_size)
            elif model_type == 'GraphFormer':
                model = GraphFormer(
                    **args, seq_len=timepoints, K=2)

            train(model, train_dataloader,
                  val_dataloader, timepoints, dataset_combination)


def train(model, train_dataloader, val_dataloader, timepoints, dataset_combination=None, configs=configs):

    with wandb.init(project='EEG-Transformers 11.0 Graphformer + ConformerCopy'):
        if dataset_combination:
            configs.update({'model': model.__class__.__name__,
                            'window_size': dataset_combination[0], 'stride': dataset_combination[1], 'dataset_strategy': dataset_combination[2], 'sequence length': timepoints})
        else:
            configs.update({'model': model.__class__.__name__,
                           'sequence length': timepoints})
        wandb.config.update(configs)
        run_name = f"{model.__class__.__name__} "
        if dataset_combination:
            window_size, stride, strategy = dataset_combination
            run_name += f" w: {window_size} s: {stride} strategy: {strategy}"
        else:
            run_name += 'continuous'
        wandb.run.name = run_name
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(
        ), lr=configs['train_params']['lr'], weight_decay=configs['train_params']['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=1, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=configs['train_params']['epochs'])
        trainer = MultiLabelClassifierTrainer(
            model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device)
        trainer.fit(epochs=configs['train_params']['epochs'])


# %%
if __name__ == '__main__':
    run()


# %%
