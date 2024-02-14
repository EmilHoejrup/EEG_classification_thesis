import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicMLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden = hidden_units
        self.fc1 = nn.Linear(in_shape, hidden_units)
        self.fc2 = nn.Linear(hidden_units, out_shape)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc2(F.relu(self.fc1(x)))
