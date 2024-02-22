# %%
import math
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
import yaml
import numpy as np
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 32
CHANNELS = 63
TIMEPOINTS = 300
INPUT_SIZE = CHANNELS * TIMEPOINTS
has_gpu = torch.cuda.is_available()
device = 'cpu'
EPOCHS = 300
# downsampling with moving average
# Hamming window?


# %%
# model = BasicMLP2(in_shape=INPUT_SIZE, out_shape=1, hidden_units=2000)
# running training loop with timer
# model = ImprovedMLP(in_shape=INPUT_SIZE, out_shape=1,
#                     hidden_units=200, num_layers=100)
# model = TinyVGG(input_shape=63, hidden_units=100, output_shape=1)
# model = NewBasic(n_channels=63, n_timepoints=300, hidden_units=30)

train_dataset = HimOrHer(train=True)
val_dataset = HimOrHer(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
model = SimpleTransformerModelVanilla(input_dim=63, d_model=300)
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()
# %%
train_dataset = DiscretizedHimOrHer(train=True)
val_dataset = DiscretizedHimOrHer(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
model = SimpleTransformerModel(input_dim=63, d_model=63, nhead=3)
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()
# %%
x, y = next(iter(train_dataloader))
x[1][1], y[1]


# %%

model = TinyVGG(input_shape=63, hidden_units=100, output_shape=1)

loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()
# %%
p_length = configs['him_or_her']['window_size']
values = [0, 1]
permutations = [list(perm)for perm in product(values, repeat=p_length)]
# %%
# %%

# Example usage
input_dim = 100   # Example vocabulary size
d_model = 256     # Example embedding size
batch_size = 32
sequence_length = 10

# Instantiate the embedding layer
embedding_layer = nn.Embedding(input_dim, d_model)

# Example input tensor
input_tensor = torch.randint(0, input_dim, (batch_size, sequence_length))
print(input_tensor.shape)
# Pass input tensor through the embedding layer
embedded_tensor = embedding_layer(input_tensor)

print(embedded_tensor.shape)  # Output: torch.Size([32, 10, 256])

# %%


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim=63, d_model=30, nhead=3, num_layers=2, dropout=0.2, max_len=5000):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        self.input_embedding = nn.Linear(input_dim, d_model)
        # self.input_embedding = nn.Embedding(input_dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, 1)

        # self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # Reshape input tensor to (timepoints, batch_size, channels)
        x = x.permute(2, 0, 1)  # Shape: (timepoints, batch_size, channels)
        print(x.shape)

        x = self.input_embedding(x)

        # x = x + self.positional_encoding[:x.size(0), :]
        print(x.shape)
        x = self.transformer_encoder(x)
        print(x.shape)

        x = x[-1, :, :]

        x = self.decoder(x)

        return x.squeeze()


# %%
m = SimpleTransformerModel()
# %%
m(x)
# %%

# Example tensor shape: [batch, channels, timepoints]
batch_size = 32
channels = 63
timepoints = 148

# Example input tensor
input_tensor = torch.randint(0, 100, (batch_size, channels, timepoints))

# Reshape the tensor to [batch * channels, timepoints]
reshaped_tensor = input_tensor.permute(
    0, 1, 2).contiguous().view(-1, timepoints)

# Define embedding dimension
embedding_dim = 20

# Instantiate the embedding layer
embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=embedding_dim)

# Pass reshaped tensor through the embedding layer
embedded_tensor = embedding_layer(reshaped_tensor)
print(embedded_tensor.shape)
# Reshape the embedded tensor back to the original shape
embedded_tensor = embedded_tensor.view(
    batch_size, channels, timepoints, embedding_dim).permute(0, 1, 3, 2)

print(embedded_tensor.shape)  # Output: torch.Size([32, 10, 256, 100])

# %%

# Example EEG data shape: [batch, channels, numericalized_timepoints]
batch_size = 32
num_channels = 63
num_timepoints = 148
embedding_dim = 20

# Example numericalized timepoints
numericalized_timepoints = torch.randint(
    0, 64, (batch_size, num_channels, num_timepoints))

# Define embedding layer
embedding_layer = nn.Embedding(num_embeddings=64, embedding_dim=embedding_dim)

# Reshape numericalized timepoints to [batch * channels * num_timepoints]
reshaped_timepoints = numericalized_timepoints.view(-1, num_timepoints)

# Embed timepoints
embedded_timepoints = embedding_layer(reshaped_timepoints)

# Reshape embedded timepoints back to [batch, channels, num_timepoints, embedding_dim]
embedded_timepoints = embedded_timepoints.view(
    batch_size, num_channels, num_timepoints, embedding_dim)

# Sum the embeddings over the timepoints dimension
# You can use other aggregation methods as well
embedded_timepoints = embedded_timepoints.sum(dim=3)

print(embedded_timepoints.shape)  # Output: torch.Size([32, 10, 256])

# %%
# %%
train_dataset = BNCI2014_001(train=True)
val_dataset = BNCI2014_001(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
x, y = next(iter(train_dataloader))
x[1][1], y[1]
# %%
# model = SimpleTransformerModelB(input_dim=22, d_model=22, nhead=2)
# model(x)
model = SimpleTransformerModelVanilla(input_dim=22)
# %%
loss_fun = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
trainer = BinaryClassifierTrainer(model, train_loader=train_dataloader,
                                  val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=20)
trainer.plot_train_val_scores()

# %%
model
# %%


class SimpleTransformerModelB(nn.Module):
    def __init__(self, input_dim=63, d_model=64, nhead=4, num_layers=2, dropout=0.2, max_len=5000):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        # self.input_embedding = nn.Linear(input_dim, d_model)
        self.embedding_layer = nn.Embedding(
            num_embeddings=64, embedding_dim=64)
        # self.input_embedding = nn.Embedding(input_dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, 1)

        # self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # Reshape input tensor to (timepoints, batch_size, channels)
        # print(x.shape)

        x = x.view(-1, 499)
        x = self.embedding_layer(x)
        print(x.shape)
        x = x.view(32, 22, 499, 64)
        x = x.sum(dim=2)
        x = x.permute(2, 0, 1)  # Shape: (timepoints, batch_size, channels)
        # print(x.shape)
        # print(x[0][0])
        # x = self.input_embedding(x)

        # x = x + self.positional_encoding[:x.size(0), :]
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)

        x = x[-1, :, :]

        x = self.decoder(x)

        return x.squeeze()

    def get_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(
            0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        return pe

# %%
