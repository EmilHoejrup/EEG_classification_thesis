
import torch.nn as nn
from einops import rearrange
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math
from torch_geometric_temporal import GraphConstructor
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.utils import dense_to_sparse


class GraphFormer(nn.Module):
    def __init__(self, vocab_size, seq_len, n_graph_features, channels, K, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5):
        super(GraphFormer, self).__init__()
        self.graph_conv = _GraphConvolution(
            seq_len, n_graph_features, channels, K, dropout)
        self.spatial_conv = nn.Conv2d(
            emb_size, emb_size, (channels, 1), (1, 1))
        # self.spatial_embedding = _SpatialEmbedding(emb_size, vocab_size)
        self.patch_embed = PatchEmbedding(emb_size)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(num_classes)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.graph_conv(x)
        x = x.unsqueeze(dim=1)
        # print("Embedding shape: ", x.shape)
        # x = rearrange(x, 'b c t e -> b e c t')
        # print("Rearranged shape: ", x.shape)
        x = self.patch_embed(x)
        # x = self.spatial_embedding(x)
        print("Spatial emb size shape: ", x.shape)
        # x = self.spatial_conv(x)
        # print("Spatial conv shape: ", x.shape)
        x = x.squeeze(dim=2)
        print("Squeezed shape: ", x.shape)
        # x = rearrange(x, 'b e t -> b t e')
        print("Rearranged shape: ", x.shape)
        x = self.transformer_encoder(x)
        print("Transformer shape: ", x.shape)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out


class _GraphConvolution(nn.Module):
    def __init__(self, seq_len, out_features, channels, K, dropout=0.5):
        super(_GraphConvolution, self).__init__()
        self.in_features = seq_len
        self.out_features = out_features
        self.channels = channels
        self.K = K
        self.graph_constructor = GraphConstructor(
            nnodes=channels, k=K, dim=seq_len, alpha=0.1)
        self.graph_conv = SGConv(seq_len, out_features, K=K)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        idx = torch.arange(self.channels).long()
        A = self.graph_constructor(idx)
        A, attr = dense_to_sparse(A)
        out = self.graph_conv(x, A, attr)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout)
        return out


# class _SpatialEmbedding(nn.Module):
#     def __init__(self, emb_size, vocab_size):
#         super(_SpatialEmbedding, self).__init__()
#         self.spatial = nn.Sequential(
#             nn.Conv2d(1, emb_size, (1, emb_size), (1, 1)),
#             nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1)),
#             nn.BatchNorm2d(emb_size),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.AvgPool2d((1, 15), (1, 5))
#         )

#     def forward(self, x):
#         return self.spatial(x)
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            # transpose, conv could enhance fiting ability slightly
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(_PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2)
                             * -(math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(_MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b t (h d) -> b h t d', h=self.nhead), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # if mask is not None:
        #     dots.masked_fill_(mask, float('-inf'))
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.fc(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, nhead=2, expansion=4, dropout=0.1):
        super().__init__(*[
            _TransformerEncoderBlock(emb_size, nhead, dropout=dropout)
            for _ in range(depth)
        ])


class _TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, nhead=2, expansion=4, dropout=0.5):
        super().__init__()
        self.attn = _MultiHeadAttention(emb_size, nhead, dropout=dropout)
        self.ff = _PositionwiseFeedforward(
            emb_size, expansion=expansion, dropout=dropout)

    def forward(self, x):
        x = _ResidualAdd(self.attn)(x)
        x = _ResidualAdd(self.ff)(x)
        return x


class _PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_size, expansion, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x):
        return self.net(x)


# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size, n_classes):
#         super().__init__()

#         # # global average pooling
#         # self.clshead = nn.Sequential(
#         #     Reduce('b n e -> b e', reduction='mean'),
#         #     nn.LayerNorm(emb_size),
#         #     nn.Linear(emb_size, n_classes)
#         # )
#         self.fc = nn.Sequential(
#             nn.Linear(emb_size, 256),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 32),
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, n_classes)
#         )

#     def forward(self, x):
#         x = x.contiguous().view(x.size(0), -1)
#         out = self.fc(x)
#         return out


class ClassificationHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(0, 256),  # Placeholder for dynamic input size
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # Compute the input size dynamically
        input_size = x.size(1) * x.size(2)

        # Update the first linear layer with the dynamic input size
        self.fc[0] = nn.Linear(input_size, 256)

        # Flatten the input
        x = x.view(x.size(0), -1)

        # Forward pass through the network
        out = self.fc(x)
        return out
