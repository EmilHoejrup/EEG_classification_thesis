# %%
from models import BasicMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import HimOrHer
import yaml
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from tqdm import tqdm
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 50


# %%
def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fun: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    # Put data into training mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)

        # Calculate loss (per batch)

        loss = loss_fun(y_logits, y)
        y_pred_labels = torch.round(y_logits)
        train_loss += loss
        # go from logits -> prediction labels
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

        # Optimizer zero_grad
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f}")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fun: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Move data to target device
            X, y = X.to(device), y.to(device)
            # Forward pass
            y_pred = model(X)

            # Calculate metrics
            test_loss += loss_fun(F.softmax(y_pred), y)
            # Go from logits -> prediction labels
            test_acc += accuracy_fn(y, y_pred)

        # Calculate test loss and accuracy average per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


train_dataset = HimOrHer(train=True)
val_dataset = HimOrHer(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE,  shuffle=True)

# running training loop with timer
model = BasicMLP(in_shape=INPUT_SIZE, out_shape=1, hidden_units=2000)
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1)

for epoch in tqdm(range(EPOCHS)):
    train_step(model=model, data_loader=train_dataloader, loss_fun=loss_fun,
               optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(model=model, data_loader=val_dataloader,
              loss_fun=loss_fun, accuracy_fn=accuracy_fn, device=device)

# %%
