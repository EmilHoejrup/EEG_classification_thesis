import math
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class BasicTransformer(nn.Module):
    def __init__(self, n_channels, seq_len, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.encoder_layer = nn.TransformerEncoderLayer(dropout=dropout,
                                                        d_model=n_channels, nhead=2, dim_feedforward=128, batch_first=True)
        self.head = SimpleClassificationHead(
            d_model=n_channels, seq_len=seq_len)

    def forward(self, x):
        b, c, t = x.shape
        x = x.view(b, t, c)
        # print(x.shape)
        x = self.encoder_layer(x)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)
        return x


class SimpleClassificationHead(nn.Module):
    def __init__(self, d_model, seq_len, hidden1=32, hidden2=8, n_classes: int = 2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # self.flatten = nn.Flatten()
        # self.seq = nn.Sequential(nn.Flatten(), nn.Linear(d_model * seq_len, hidden1), nn.ReLU(), nn.Dropout(dropout), nn.Linear(
        #     hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden2, n_classes))
        self.seq = nn.Sequential(
            nn.Flatten(), nn.Linear(d_model*seq_len, n_classes))

    def forward(self, x):

        x = self.norm(x)

        x = self.seq(x)

        return x


class ConTransformer(nn.Module):

    def __init__(self,  d_model=100, n_head=4, max_len=1225, seq_len=200,
                 ffn_hidden=128, n_layers=2, drop_prob=0.1, details=False, n_channels=22, n_classes=2):
        super().__init__()
        self.details = details
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.encoder_input_layer = nn.Linear(
            in_features=n_channels,
            out_features=d_model
        )

        self.pos_emb = PostionalEncoding(
            max_seq_len=max_len, batch_first=False, d_model=d_model, dropout=0.1)
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               details=details,)
        self.classHead = SimpleClassificationHead(
            seq_len=seq_len, d_model=d_model, n_classes=n_classes)

    def forward(self, src):
        # src = torch.reshape(src, (-1, self.seq_len, self.n_channels))
        # if self.details:
        #     print('before input layer: ' + str(src.size()))
        # src = self.encoder_input_layer(src)
        # if self.details:
        #     print('after input layer: ' + str(src.size()))
        # src = self.pos_emb(src)
        # if self.details:
        #     print('after pos_emb: ' + str(src.size()))
        # enc_src = self.encoder(src)
        # cls_res = self.classHead(enc_src)
        # if self.details:
        #     print('after cls_res: ' + str(cls_res.size()))
        # return cls_res
        return self.classHead(self.encoder(self.pos_emb(self.encoder_input_layer(torch.reshape(src, (-1, self.seq_len, self.n_channels))))))


class NewTransformer(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=30, seq_len=8228, dropout=0.1, dim_ff=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.emb = nn.Embedding(self.vocab_size, embedding_dim=embedding_dim)
        # self.class_head = ClassificationHead(hidden1=128,hidden2=64,hidden3=32,
        #     d_model=embedding_dim, seq_len=seq_len, n_classes=2, details=False)
        self.class_head = SimpleClassificationHead(
            d_model=embedding_dim, seq_len=seq_len, dropout=dropout)
        # self.positional = PostionalEncoding(d_model=seq_len, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        batch, channels, timepoints = x.shape
        x = x.to(torch.long)
        x = x.view(batch, -1)

        # x = self.positional(x)
        print(x.shape)
        # print(x.dtype)
        x = self.emb(x)
        # x = F.one_hot(x)
        print(x.shape)
        # x = x.to(torch.float32)
        # print(x.dtype)
        x = self.layer_norm(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.encoder_layer(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.class_head(x)
        return x


class TransWithEmbeddingV1(nn.Module):
    def __init__(self,  d_model, nhead, num_layers, vocab_size, emb_dim):
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, 2)

    def forward(self, x):
        # Reshape input tensor to (timepoints, batch_size, channels)
        # print(x.shape)

        batch_size, channels, timepoints = x.shape

        x = x.view(-1, timepoints)
        x = self.embedding_layer(x)
        # print(x.shape)
        x = x.view(batch_size, channels,
                   timepoints, self.emb_dim)
        x = x.sum(dim=2)
        x = x.permute(2, 0, 1)  # Shape: (timepoints, batch_size, channels)
        # print(x.shape)

        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)

        x = x[-1, :, :]

        x = self.decoder(x)

        return x.squeeze()


class SimpleTransformerModel(nn.Module):
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
        batch_size, channels, timepoints = x.shape

        x = x.view(-1, timepoints)
        x = self.embedding_layer(x)
        x = x.view(batch_size, channels, timepoints, self.d_model)
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


class SimpleTransformerModelVanilla(nn.Module):
    def __init__(self, input_dim=63, d_model=300, nhead=4, num_layers=2, dropout=0.2, max_len=5000):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        self.input_embedding = nn.Linear(input_dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, 2)

        self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # Reshape input tensor to (timepoints, batch_size, channels)
        x = x.permute(2, 0, 1)  # Shape: (timepoints, batch_size, channels)

        x = self.input_embedding(x)

        x = x + self.positional_encoding[:x.size(0), :]

        x = self.transformer_encoder(x)

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


class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(
            1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.bn1 = nn.BatchNorm2d(
            16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(
            2, 1), stride=(1, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(
            32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pooling = nn.AvgPool2d(
            kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(
            1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.bn3 = nn.BatchNorm2d(
            32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(
            1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(
            64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = nn.Conv2d(64, 2, kernel_size=(
            1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(
            2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.mean(x, dim=(2, 3))  # Global average pooling
        print(x.shape)
        print(x)

        # Apply softmax
        x = self.softmax(x)
        # x = x[:, 0]
        # Get class predictions by taking the index of the maximum value

        return x

# FROM HUGGINGFACE INTRO TO NLP TRANSFORMERS


# def scaled_dot_product_attention(query, key, value):
#     dim_k = query.size(-1)
#     scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
#     weights = F.softmax(scores, dim=1)
#     return torch.bmm(weights, value)


# class AttentionHead(nn.Module):
#     def __init__(self, embed_dim, head_dim):
#         super().__init__()
#         self.q = nn.Linear(embed_dim, head_dim)
#         self.k = nn.Linear(embed_dim, head_dim)
#         self.v = nn.Linear(embed_dim, head_dim)
#         print(f"q,k,v shape: {embed_dim, head_dim}")

#     def forward(self, hidden_state):
#         x = scaled_dot_product_attention(
#             self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
#         print(f"scaled_dot_product_attention: {x.shape}")
#         return x


# class MultiHeadAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         embed_dim = config['hidden_size']
#         num_heads = config['num_attention_heads']
#         head_dim = embed_dim // num_heads
#         self.heads = nn.ModuleList(
#             [AttentionHead(embed_dim=embed_dim, head_dim=head_dim)
#              for _ in range(num_heads)]
#         )
#         self.output_linear = nn.Linear(embed_dim, embed_dim)

#     def forward(self, hidden_state):
#         x = torch.cat([head(hidden_state) for head in self.heads], dim=1)
#         print(f"multihead attention before output layer: {x.shape}")
#         return self.output_linear(x)


# class BasicMLP(nn.Module):
#     def __init__(self, in_shape, out_shape, hidden_units):
#         super().__init__()
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.hidden = hidden_units
#         self.fc1 = nn.Linear(in_shape, hidden_units)
#         self.fc2 = nn.Linear(hidden_units, out_shape)
#         # Perform Xavier initialization on the weights
#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         return self.fc2(F.relu(self.fc1(x)))


# class NewBasic(nn.Module):
#     def __init__(self, n_channels, n_timepoints, hidden_units):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.conv1 = nn.Conv1d(
#             in_channels=n_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
#         self.l1 = nn.Linear(2400, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         # print(x.shape)
#         x = F.relu(self.flatten(x))
#         # print(x.shape)
#         x = F.relu(self.l1(x))
#         # print(x.shape)
#         x = self.flatten(x)
#         x = x.squeeze(-1)
#         return x


# class BasicMLP2(nn.Module):
#     def __init__(self, in_shape, out_shape, hidden_units):
#         super().__init__()
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.hidden = hidden_units
#         self.block1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=in_shape, out_features=hidden_units),
#             nn.ReLU(),
#             nn.Linear(hidden_units, hidden_units),
#             nn.ReLU(),
#             nn.Linear(hidden_units, out_shape),
#         )

#     def forward(self, x):
#         # batch_size = x.size(0)
#         # x = x.view(batch_size, -1)
#         return self.block1(x).squeeze()


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
