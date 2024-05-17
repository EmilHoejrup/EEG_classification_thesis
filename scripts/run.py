# %%
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from datasets import *
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from support.utils import test_metrics
from braindecode.models import ShallowFBCSPNet, EEGConformer
from torch.optim.lr_scheduler import LRScheduler
from itertools import product
from trainer import Trainer
from models import *
from GraphFormer import GraphFormer

has_gpu = torch.cuda.is_available()
device = 'mps' if getattr(
    torch, 'torch.backends.mps.is_built()', False) else 'cuda' if has_gpu else 'cpu'


def run():
    with open(CONFIG_FILE, 'r') as file:
        configs = yaml.safe_load(file)
    if configs['train_params']['wandb_logging']:
        wandb.login()
    datasets = configs['dataset_list']
    for dataset_name in datasets:
        dataset = globals()[dataset_name]
        dataset_params = configs.get('datasets').get(dataset_name)
        train_params = configs.get('train_params')

        if dataset_params == None:
            train_dataset = dataset(train=True)
            val_dataset = dataset(train=False, val=True)
            test_dataset = dataset(train=False, val=False, test=True)
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
            test_dataloader = DataLoader(
                dataset=test_dataset, batch_size=train_params['batch_size'], shuffle=True)
            _, _, timepoints = train_dataset.get_X_shape()
            train_models(train_dataloader, val_dataloader, test_dataloader,
                         timepoints)
        else:
            param_combinations = product(*dataset_params.values())
            for combination in param_combinations:
                train_dataset = dataset(*combination, train=True)
                val_dataset = dataset(*combination, train=False, val=True)
                test_dataset = dataset(
                    *combination, train=False, val=False, test=True)
                train_dataloader = DataLoader(
                    dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
                val_dataloader = DataLoader(
                    dataset=val_dataset, batch_size=train_params['batch_size'], shuffle=True)
                test_dataloader = DataLoader(
                    dataset=test_dataset, batch_size=train_params['batch_size'], shuffle=True)
                _, _, timepoints = train_dataset.get_X_shape()
                vocab_size = train_dataset.get_vocab_size()
                train_models(train_dataloader, val_dataloader, test_dataloader,
                             timepoints, combination, vocab_size=vocab_size)


def train_models(train_dataloader, val_dataloader, test_dataloader, timepoints, dataset_combination=None, configs=configs, vocab_size=None):
    models = configs['models_to_train']
    for model_type in models:
        model_params = configs.get('models').get(model_type)
        # print(model_params)
        param_combinations = product(*model_params.values())

        for combination in param_combinations:
            num_classes = len(np.unique(train_dataloader.dataset.Y_train))
            param_keys = model_params.keys()
            args = dict(zip(param_keys, combination))
            if model_type == 'EEGConformer':
                model = EEGConformer(
                    **args, n_outputs=num_classes, n_times=timepoints, sfreq=250)
            elif model_type == 'PPModel':
                model = PPModel(
                    **args, seq_len=timepoints, num_classes=num_classes, vocab_size=vocab_size)
            elif model_type == 'TransformerOnly':
                model = TransformerOnly(
                    **args, seq_len=timepoints, num_classes=num_classes, vocab_size=vocab_size)
            elif model_type == 'EEGTransformerEmb':
                model = EEGTransformerEmb(
                    **args, seq_len=timepoints, num_classes=num_classes, vocab_size=vocab_size)
            elif model_type == 'GraphFormer':
                model = GraphFormer(
                    **args, num_classes=num_classes, seq_len=timepoints)
            elif model_type == 'SimplePPModel':
                model = SimplePPModel(
                    **args, num_classes=num_classes, vocab_size=vocab_size)
            elif model_type == 'SimpleGraphFormer':
                model = SimpleGraphFormer(
                    **args, num_classes=num_classes, seq_len=timepoints)
            elif model_type == 'ShallowFBCSPNet':
                model = ShallowFBCSPNet(
                    **args, n_classes=num_classes, input_window_samples=timepoints, sfreq=250)
            elif model_type == 'SimpleShallowNet':
                model = SimpleShallowNet(
                    **args, num_classes=num_classes, timepoints=timepoints)
            elif model_type == 'SimpleConformer':
                model = SimpleConformer(
                    **args, num_classes=num_classes, timepoints=timepoints)

            train(model, train_dataloader,
                  val_dataloader, test_dataloader, timepoints, dataset_combination, configs=configs, args=args)


def train(model, train_dataloader, val_dataloader, test_dataloader, timepoints, dataset_combination=None, configs=configs, args=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(
    ), lr=configs['train_params']['lr'], weight_decay=configs['train_params']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=configs['train_params']['epochs'])
    if configs['train_params']['wandb_logging']:
        with wandb.init(project=configs['train_params']['project_name']):
            if dataset_combination:
                configs.update({'model': model.__class__.__name__,
                                'window_size': dataset_combination[0], 'stride': dataset_combination[1],  'sequence length': timepoints})
            else:
                configs.update({'model': model.__class__.__name__,
                                'sequence length': timepoints})
            wandb.config.update(configs)
            wandb.config.update(args)
            run_name = f"{model.__class__.__name__} {train_dataloader.dataset.__class__.__name__}"
            if dataset_combination:
                window_size, stride = dataset_combination
                run_name += f" w: {window_size} s: {stride}"
            else:
                run_name += 'continuous'
            run_name += configs['train_params']['run_name']
            wandb.run.name = run_name

            trainer = Trainer(
                model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device)
            trainer.fit(epochs=configs['train_params']
                        ['epochs'], print_metrics=False)
            if configs['test']:
                test_accuracy, kappa, precision, recall = test_metrics(
                    model, test_dataloader)
                wandb.log({'Test accuracy': test_accuracy, 'Kappa': kappa,
                          'Precision': precision, 'Recall': recall})
                wandb.log({'F1 Score': 2*(precision*recall)/(precision+recall)})
    else:
        trainer = Trainer(
            model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, wandb_logging=False)
        trainer.fit(epochs=configs['train_params']
                    ['epochs'], print_metrics=False)

        if configs['test']:
            test_accuracy, kappa, precision, recall = test_metrics(
                model, test_dataloader)
            print(f"Test accuracy: {test_accuracy}")
            print(f"Kappa: {kappa}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")


# %%
if __name__ == '__main__':
    run()


# %%
