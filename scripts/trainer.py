import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from support.utils import plot_train_val_scores
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_score, recall_score


class Trainer(nn.Module):
    """
    A class for training and evaluating a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to train and evaluate.
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        loss_fun (callable): The loss function to use for training and evaluation.
        optimizer (Optimizer): The optimizer to use for training.
        scheduler (LRScheduler): The learning rate scheduler.
        device (str or torch.device): The device to use for training and evaluation.
        wandb_logging (bool, optional): Whether to log metrics to Weights & Biases. Defaults to True.
    """

    def __init__(self, model, train_loader, val_loader, loss_fun, optimizer, scheduler, device, wandb_logging=True):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.scheduler = scheduler
        self.model.to(device)
        self.wandb_logging = wandb_logging
        if wandb_logging:
            wandb.watch(self.model, self.loss_fun, log='all')

    def fit(self, epochs=5, print_metrics=False):
        """
        Trains the model for the specified number of epochs.

        Args:
            epochs (int, optional): The number of epochs to train the model. Defaults to 5.
            print_metrics (bool, optional): Whether to print metrics during training. Defaults to False.
        """
        for _ in tqdm(range(epochs)):
            self._train_step(print_metrics)
            self._val_step(print_metrics)

    def get_metrics(self):
        """
        Returns the training and validation metrics.

        Returns:
            tuple: A tuple containing the training losses, training accuracies, validation losses, and validation accuracies.
        """
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def plot_train_val_scores(self):
        """
        Plots the training and validation scores.
        """
        with torch.inference_mode():
            plot_train_val_scores(
                self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies)

    def _train_step(self, print_metrics):
        """
        Performs a single training step.

        Args:
            print_metrics (bool): Whether to print metrics during training.
        """
        train_loss, train_acc = 0, 0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_logits = self.model(X)

            # Calculate metrics
            loss = self.loss_fun(y_logits, y)
            train_loss += loss.item()
            _, predicted = torch.max(y_logits, 1)
            train_acc += (predicted == y).sum().item()
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(self.train_loader)
        train_acc /= (len(self.train_loader.dataset))

        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        if print_metrics:
            print(
                f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.5f}")
        if self.wandb_logging:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})

    def _val_step(self, print_metrics):
        """
        Performs a single validation step.

        Args:
            print_metrics (bool): Whether to print metrics during validation.
        """
        val_loss, val_acc = 0, 0
        self.model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.val_loader):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                val_loss += self.loss_fun(y_logits, y).item()
                _, predicted = torch.max(y_logits, 1)
                val_acc += (predicted == y).sum().item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

            val_loss /= len(self.val_loader)
            val_acc /= (len(self.val_loader.dataset))
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

        val_precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        val_recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        if print_metrics:
            print(f"Val   loss: {val_loss:.5f} | Val   acc: {val_acc*100:.5f}")
            print(
                f"Val Precision: {val_precision:.5f} | Val Recall: {val_recall:.5f}")

        if self.wandb_logging:
            wandb.log({"Validation Loss": val_loss,
                       "Validation Accuracy": val_acc,
                       "Validation Precision": val_precision,
                       "Validation Recall": val_recall,
                       "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred)
                       })
