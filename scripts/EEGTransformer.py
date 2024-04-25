from layers import _PositionalEncoding, _TransformerEncoder, ClassificationHead, _SpatialEmbedding
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math


from layers import _PositionalEncoding, _TransformerEncoder, ClassificationHead


class EEGTransformerFlatten(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead=2, num_classes=2, expansion=4, dropout=0.5):
        super(EEGTransformerFlatten, self).__init__()
        # self.transformer_encoder = _TransformerEncoder(
        #     depth, emb_size, nhead, expansion, dropout)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=expansion*emb_size, dropout=dropout, batch_first=True)
        self.clshead = _ClassificationHead(emb_size, n_classes=num_classes)
        self.positional_encoding = _PositionalEncoding(vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        # x = rearrange(x, 'b c t -> b t c')
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        out = self.clshead(x)
        return out


class _ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super(_ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class _ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super(_ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class EEGTransformer(nn.Module):
    def __init__(self, seq_len, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(EEGTransformer, self).__init__()
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)
        self.positional_encoding = _PositionalEncoding(emb_size)
        self.embedding = nn.Embedding(vocab_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(-1)
        x = rearrange(x, 'b c t -> b t c')
        # x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out


class EEGTransformerEmb(nn.Module):
    def __init__(self, seq_len, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5):
        super(EEGTransformerEmb, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.spatial_conv = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
        self.spatial_embedding = _SpatialEmbedding(emb_size)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.embedding(x)
        # print("Embedding shape: ", x.shape)
        x = rearrange(x, 'b c t e -> b e c t')
        # print("Rearranged shape: ", x.shape)
        # x = self.spatial_conv(x)
        x = self.spatial_embedding(x)
        # print("Spatial conv shape: ", x.shape)
        x = x.squeeze(dim=2)
        # print("Squeezed shape: ", x.shape)
        x = rearrange(x, 'b e t -> b t e')
        x = self.transformer_encoder(x)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out
