
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math
from layers import _PositionalEncoding, _TransformerEncoder, ClassificationHead, _SpatialEmbedding


class ConformerCopy(nn.Module):
    def __init__(self, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5):
        super(ConformerCopy, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        self.spatial_conv = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
        self.spatial_embedding = _SpatialEmbedding(emb_size)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(194*emb_size, num_classes)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = x.unsqueeze(dim=1)
        # print("Embedding shape: ", x.shape)
        # x = rearrange(x, 'b c t e -> b e c t')
        # print("Rearranged shape: ", x.shape)
        # x = self.spatial_conv(x)
        x = self.spatial_embedding(x)
        # print("Spatial conv shape: ", x.shape)
        x = x.squeeze(dim=2)
        # print("Squeezed shape: ", x.shape)
        x = rearrange(x, 'b e t -> b t e')
        x = self.transformer_encoder(x)
        # print("Transformer shape: ", x.shape)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out
