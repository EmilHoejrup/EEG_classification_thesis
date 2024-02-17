# %%
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from models import *
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
BATCH_SIZE = 64
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 100


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
        # X = X.unsqueeze(1)
        # Forward pass
        y_logits = model(X)

        # Calculate loss (per batch)

        loss = loss_fun(y_logits.squeeze(), y)
        train_loss += loss
        y_pred_labels = torch.round(torch.sigmoid(y_logits.squeeze()))
        # if batch % 100 == 0:
        # print(torch.sigmoid(y_logits))
        # print(y_pred_labels)

        # go from logits -> prediction labels
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred_labels)

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
        # for X, y in data_loader:
        for batch, (X, y) in enumerate(data_loader):
            # X = X.unsqueeze(1)
            # Move data to target device
            X, y = X.to(device), y.to(device)
            # Forward pass
            y_pred = model(X)
            y_pred_labels = torch.round(torch.sigmoid(y_pred.squeeze()))

            # Calculate metrics
            test_loss += loss_fun(y_pred.squeeze(), y)
            # Go from logits -> prediction labels
            test_acc += accuracy_fn(y, y_pred_labels)

        # Calculate test loss and accuracy average per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")


# full_dataset = HimOrHer()
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(
#     full_dataset, [train_size, test_size])
train_dataset = HimOrHer(train=True)
val_dataset = HimOrHer(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
# model = BasicMLP2(in_shape=INPUT_SIZE, out_shape=1, hidden_units=2000)
# running training loop with timer
# model = ImprovedMLP(in_shape=INPUT_SIZE, out_shape=1,
#                     hidden_units=200, num_layers=100)
model = TinyVGG(input_shape=63, hidden_units=100, output_shape=1)
# model = NewBasic(n_channels=63, n_timepoints=300, hidden_units=30)


loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

for epoch in tqdm(range(EPOCHS)):
    train_step(model=model, data_loader=train_dataloader, loss_fun=loss_fun,
               optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(model=model, data_loader=val_dataloader,
              loss_fun=loss_fun, accuracy_fn=accuracy_fn, device=device)


# %%
x, y = next(iter(train_dataloader))
x[0], y[0]


# %%
