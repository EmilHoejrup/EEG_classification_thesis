import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from support.utils import plot_train_val_scores


class BinaryClassifierTrainer(nn.Module):
    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 loss_fun: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device):
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

    def fit(self, epochs=5, print_metrics=True):
        for _ in tqdm(range(epochs)):
            self._train_step(print_metrics)
            self._val_step(print_metrics)

    def plot_train_val_scores(self):
        with torch.inference_mode():
            plot_train_val_scores(
                self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies)

    def _train_step(self, print_metrics):
        train_loss, train_acc = 0, 0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_logits = self.model(X)

            # Calculate metrics
            loss = self.loss_fun(y_logits, y)
            train_loss += loss
            y_labels = torch.round(torch.sigmoid(y_logits))
            train_acc += self._accuracy_fun(y, y_labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        if (print_metrics):
            print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f}")

    def _val_step(self, print_metrics):
        val_loss, val_acc = 0, 0
        self.model.eval()
        with torch.inference_mode():
            # for X, y in data_loader:
            for batch, (X, y) in enumerate(self.val_loader):
                X, y = X.to(self.device), y.to(self.device)
                # Forward pass
                y_logits = self.model(X)
                y_labels = torch.round(torch.sigmoid(y_logits))

                # Calculate metrics
                val_loss += self.loss_fun(y_logits, y)
                # Go from logits -> prediction labels
                val_acc += self._accuracy_fun(y, y_labels)

            # Calculate test loss and accuracy average per batch
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
        if (print_metrics):
            print(f"Val   loss: {val_loss:.5f} | Val   acc: {val_acc:.5f}")

    def _accuracy_fun(self, y_pred, y_true):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = correct / len(y_pred) * 100
        return acc