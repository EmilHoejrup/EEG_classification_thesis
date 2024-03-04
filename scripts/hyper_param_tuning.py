# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import BNCI_LEFT_RIGHT
from new_model import Transformer
from trainer import MultiLabelClassifierTrainer
import mlflow
from mlflow.tracking import MlflowClient
import torch.optim as optim
import yaml
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)

LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 64
has_gpu = torch.cuda.is_available()
device = 'mps' if getattr(
    torch, 'has_mps', False) else 'cuda' if has_gpu else 'cpu'
device = 'cpu'


def train(run_name, window_size, stride):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('window_size', window_size)
        mlflow.log_param('stride', stride)

        train_dataset = BNCI_LEFT_RIGHT(
            train=True, window_size=window_size, stride=stride)
        val_dataset = BNCI_LEFT_RIGHT(
            train=False, window_size=window_size, stride=stride)
        train_dataloader = DataLoader(
            dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, )
        images, channels, timepoints = train_dataset.get_X_shape()

        # Train model
        model = Transformer(device=device, seq_len=timepoints)
        model = model.to(device)
        loss_fun = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=EPOCHS - 1)
        trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                              val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
        trainer.fit(epochs=EPOCHS)
        # trainer.plot_train_val_scores()
        train_losses, train_accuracies, val_losses, val_accuracies = trainer.get_metrics()
        metrics = {'train_loss': train_losses[-1], 'train_accuracy': train_accuracies[-1],
                   'val_loss': val_losses[-1], 'val_accuracy': val_accuracies[-1]}
        mlflow.log_metrics(metrics)


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp_name = "BNCI LR dataset hyperparametertuning"
    run_name = "first test"
    mlflow.set_experiment(exp_name)

    time_window = range(4, 6)
    stride = range(2, 3)
    for tw in time_window:
        for st in stride:
            train(f"{run_name}{tw,st}", tw, st)


if __name__ == '__main__':
    main()
