from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicMLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden = hidden_units
        self.fc1 = nn.Linear(in_shape, hidden_units)
        self.fc2 = nn.Linear(hidden_units, out_shape)
        # Perform Xavier initialization on the weights
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc2(F.relu(self.fc1(x)))


class NewBasic(nn.Module):
    def __init__(self, n_channels, n_timepoints, hidden_units):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(
            in_channels=n_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(2400, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.flatten(x))
        # print(x.shape)
        x = F.relu(self.l1(x))
        # print(x.shape)
        x = self.flatten(x)
        x = x.squeeze(-1)
        return x
# Check balancen af categorier
# NORMALIsER! (z-score)
# evt normalisér pr kanal


class BasicMLP2(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden = hidden_units
        self.block1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_shape),
        )

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        return self.block1(x)


class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, kernel_size=30, stride=1, padding=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)

        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=output_shape)

        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))
        # x = self.block1(x)
        # print(x.shape)
        # x = self.block2(x)
        # print(x.shape)
        # x = nn.Flatten(x)
        # print(x.shape)
