from layers import _GraphConvolution, _PositionalEncoding, _TransformerEncoder, ClassificationHead, _SpatialEmbedding
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math


from layers import _PositionalEncoding, _TransformerEncoder, ClassificationHead


class SimpleShallowNet(nn.Module):
    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75):
        super(SimpleShallowNet, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            in_channels, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels*maxpool_out, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = F.elu(self.spatio_temporal(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SimpleConformer(nn.Module):
    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75, nhead=2):
        super(SimpleConformer, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            in_channels, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)

        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(
            self.encoder_layers, num_layers=6, norm=nn.LayerNorm(num_kernels))
        hidden1_size = (num_kernels*maxpool_out)//2
        hidden2_size = hidden1_size//2
        self.fc = nn.Sequential(
            nn.Linear(num_kernels*maxpool_out, hidden1_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2_size, num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = self.spatio_temporal(x)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b c t -> b t c')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConformerCopy(nn.Module):
    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75, nhead=2):
        super(ConformerCopy, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (in_channels, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)

        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(
            self.encoder_layers, num_layers=6, norm=nn.LayerNorm(num_kernels))
        hidden1_size = 256
        hidden2_size = 32
        self.fc = nn.Sequential(
            nn.Linear(num_kernels*maxpool_out, hidden1_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2_size, num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        # x = F.elu(self.spatio_temporal(x))
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.squeeze(dim=2)
        # x = rearrange(x, 'b c t -> b t c')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimplePPModel(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead=2, num_classes=2, expansion=4, dropout=0.5):
        super(SimplePPModel, self).__init__()
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


class TransformerOnly(nn.Module):
    def __init__(self, seq_len, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(TransformerOnly, self).__init__()
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)
        self.positional_encoding = _PositionalEncoding(emb_size)

    def forward(self, x):
        x = x.squeeze(-1)
        x = rearrange(x, 'b c t -> b t c')
        # x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out


class PPModel(nn.Module):
    def __init__(self, seq_len, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(PPModel, self).__init__()
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


class SimpleGraphFormer(nn.Module):
    def __init__(self, seq_len, K, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5, avg_pool_kernel=15, avg_pool_stride=5, num_blocks=3):
        super(SimpleGraphFormer, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        avg_pool_output = (seq_len-avg_pool_kernel)//avg_pool_stride + 1
        self.spatial_conv = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
        self.graph_conv = _GraphConvolution(
            avg_pool_output, avg_pool_output, 22, K)
        self.graph_convolutions = nn.Sequential(
            *[_GraphConvolution(seq_len, seq_len, 22, K) for _ in range(num_blocks)])
        self.spatial_embedding = _SpatialEmbedding(emb_size)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(
            avg_pool_output*emb_size, num_classes)
        self.avgpool = nn.AvgPool2d((1, avg_pool_kernel), (1, avg_pool_stride))

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        # x = x.unsqueeze(dim=1)
        # print("Embedding shape: ", x.shape)
        # x = rearrange(x, 'b c t e -> b e c t')
        # print("Rearranged shape: ", x.shape)
        # x = self.graph_conv(x)
        # x = _ResidualAdd(self.graph_conv)(x)
        x = self.graph_convolutions(x)
        x = self.avgpool(x)
        # x = self.spatial_conv(x)
        # x = self.spatial_embedding(x)
        # print("Graph conv shape: ", x.shape)
        # print("Spatial conv shape: ", x.shape)
        x = x.squeeze(dim=2)
        # print("Squeezed shape: ", x.shape)
        x = rearrange(x, 'b e t -> b t e')
        x = self.transformer_encoder(x)
        # print("Transformer shape: ", x.shape)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out
