import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from support.utils import plot_train_val_scores
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_score, recall_score


class Trainer(nn.Module):
    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 loss_fun: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: LRScheduler,
                 device: torch.device,
                 wandb_logging=True):

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

    def fit(self, epochs=5, print_metrics=True):
        for _ in tqdm(range(epochs)):
            self._train_step(print_metrics)
            self._val_step(print_metrics)

    def get_metrics(self):
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def plot_train_val_scores(self):
        with torch.inference_mode():
            plot_train_val_scores(
                self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies)

    def _train_step(self, print_metrics):
        train_loss, train_acc = 0, 0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            # X = X.unsqueeze(dim=2)

            # Forward pass
            # X = X.reshape()
            # X = torch.transpose(X, 1, 2)
            # print(X.shape)
            # X = torch.reshape(X, (-1, 562, 22))
            y_logits = self.model(X)

            # Calculate metrics
            loss = self.loss_fun(y_logits, y)
            train_loss += loss.item()
            train_acc += (y_logits.argmax(1) == y).sum().item()
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(self.train_loader)
        train_acc /= (len(self.train_loader.dataset))

        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        if (print_metrics):
            print(
                f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.5f}")
        if self.wandb_logging:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})

    def _val_step(self, print_metrics):
        val_loss, val_acc = 0, 0
        val_precision, val_recall = 0, 0
        self.model.eval()
        with torch.inference_mode():
            y_true, y_pred = [], []
            for batch, (X, y) in enumerate(self.val_loader):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                val_acc += (y_logits.argmax(1) ==
                            y).type(torch.float).sum().item()

                # Calculate metrics
                val_loss += self.loss_fun(y_logits, y)
                # Go from logits -> prediction labels
                y_pred.extend(y_logits.argmax(1).cpu().numpy())
                y_true.extend(y.cpu().numpy())

            # Calculate test loss and accuracy average per batch
            val_loss /= len(self.val_loader)
            val_acc /= (len(self.val_loader.dataset))
            # self.log('validation loss', val_loss)
            # self.log('validation accuracy', val_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

        val_precision = precision_score(y_true,
                                        y_pred, average='binary', zero_division=0)
        val_recall = recall_score(y_true, y_pred, average='binary',
                                  zero_division=0)
        if (print_metrics):
            print(f"Val   loss: {val_loss:.5f} | Val   acc: {val_acc*100:.5f}")
            print(
                f"Val Precision: {val_precision:.5f} | Val Recall: {val_recall:.5f}")
            print(f"y_true: {y_true}")
            print(f"y_pred: {y_pred}")

        if self.wandb_logging:
            wandb.log({"Validation Loss": val_loss,
                       "Validation Accuracy": val_acc,
                       "Validation Precision": val_precision,
                       "Validation Recall": val_recall,
                       "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred)})
