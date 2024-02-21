import math
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim=63, d_model=64, nhead=4, num_layers=2, dropout=0.2, max_len=5000):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        # self.input_embedding = nn.Linear(input_dim, d_model)
        self.embedding_layer = nn.Embedding(
            num_embeddings=64, embedding_dim=20)
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

        x = x.view(-1, 148)
        x = self.embedding_layer(x)
        x = x.view(32, 63, 148, 20)
        x = x.sum(dim=3)
        x = x.permute(2, 0, 1)  # Shape: (timepoints, batch_size, channels)
        # print(x.shape)
        print(x[0][0])
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


class VanillaTransformerModel(nn.Module):
    # 148
    def __init__(self, input_dim=63, d_model=300, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x.squeeze()

# FROM HUGGINGFACE INTRO TO NLP TRANSFORMERS


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
    weights = F.softmax(scores, dim=1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        print(f"q,k,v shape: {embed_dim, head_dim}")

    def forward(self, hidden_state):
        x = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        print(f"scaled_dot_product_attention: {x.shape}")
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config['hidden_size']
        num_heads = config['num_attention_heads']
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim=embed_dim, head_dim=head_dim)
             for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([head(hidden_state) for head in self.heads], dim=1)
        print(f"multihead attention before output layer: {x.shape}")
        return self.output_linear(x)


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
        return self.block1(x).squeeze()


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
        x = self.classifier(self.block2(self.block1(x)))
        return x.squeeze()
