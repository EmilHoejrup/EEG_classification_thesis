# from layers import _GraphConvolution, _SpatialEmbedding, _TransformerEncoder, ClassificationHead
# from einops import rearrange
# import torch
# from torch import nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange, Reduce
# import math
# from torch_geometric.nn import GCNConv, SGConv
# from torch_geometric.utils import dense_to_sparse


from layers import _GraphConvolution, _ResidualAdd, _SpatialEmbedding, _TransformerEncoder, ClassificationHead
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.utils import dense_to_sparse

#### GRAPHFORMER LAST LAYER ####

# class GraphFormer(nn.Module):
#     def __init__(self, seq_len, K=2, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5):
#         super(GraphFormer, self).__init__()
#         # self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.spatial_conv = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
#         self.graph_conv = _GraphConvolution(196, 196, emb_size, K)
#         self.spatial_embedding = _SpatialEmbedding(emb_size)
#         self.transformer_encoder = _TransformerEncoder(
#             depth, emb_size, nhead, expansion, dropout)
#         self.clshead = ClassificationHead(196*emb_size, num_classes)

#     def forward(self, x):
#         # x = torch.unsqueeze(x, dim=1)
#         x = x.unsqueeze(dim=1)
#         # print("Embedding shape: ", x.shape)
#         # x = rearrange(x, 'b c t e -> b e c t')
#         # print("Rearranged shape: ", x.shape)
#         # x = self.spatial_conv(x)
#         x = self.spatial_embedding(x)
#         # print("Spatial conv shape: ", x.shape)
#         x = x.squeeze(dim=2)
#         x = self.graph_conv(x)
#         # print("Squeezed shape: ", x.shape)
#         x = rearrange(x, 'b e t -> b t e')
#         x = self.transformer_encoder(x)
#         # print("Transformer shape: ", x.shape)
#         # x, out = self.clshead(x)
#         out = self.clshead(x)
#         return out

#### TEMPORAL GRAPH ####


class _SpatialEmbedding(nn.Module):

    def __init__(self, emb_size, K, seq_len, avg_pool_kernel, avg_pool_stride):
        super(_SpatialEmbedding, self).__init__()
        self.temporal_graph = TemporalGraph(seq_len, K, emb_size)
        self.spatial = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
        self.batchnorm = nn.BatchNorm2d(emb_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.temporal_conv = nn.Conv2d(1, 1, (1, 25), (1, 1))

        self.avgpool = nn.AvgPool2d((1, avg_pool_kernel), (1, avg_pool_stride))

    def forward(self, x):
        # x = self.temporal(x)
        # print("Temporal shape: ", x.shape)
        # x = rearrange(x, 'b e c t -> b e (c t)')
        print("Input shape: ", x.shape)
        x = self.temporal_conv(x)
        print("Temporal conv shape: ", x.shape)
        x = self.temporal_graph(x)

        # print("Graph conv shape: ", x.shape)
        x = self.spatial(x)
        # print("Spatial shape: ", x.shape)
        x = self.batchnorm(x)
        # print("Batchnorm shape: ", x.shape)
        x = self.elu(x)
        # print("ELU shape: ", x.shape)
        x = self.dropout(x)
        # print("Dropout shape: ", x.shape)
        x = self.avgpool(x)
        # print("Avgpool shape: ", x.shape)

        return x


class TemporalGraph(nn.Module):
    def __init__(self, seq_len, K, emb_size):
        super(TemporalGraph, self).__init__()
        self.graph_conv = _GraphConvolution(seq_len, seq_len, 22, K)
        self.emb_size = emb_size

    def forward(self, x):
        layer_outputs = []
        for _ in range(self.emb_size):
            output = self.graph_conv(x)
            layer_outputs.append(output)
        x = torch.stack(layer_outputs, dim=1)
        return x


class GraphFormer(nn.Module):
    def __init__(self, seq_len, K=2, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5, avg_pool_kernel=5, avg_pool_stride=3):
        super(GraphFormer, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_size)

        self.spatial_embedding = _SpatialEmbedding(
            emb_size, K, seq_len, avg_pool_kernel, avg_pool_stride)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        avg_pool_output = (seq_len-avg_pool_kernel)//avg_pool_stride + 1
        self.clshead = ClassificationHead(
            avg_pool_output*emb_size, num_classes)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        # x = x.unsqueeze(dim=1)
        # print("Embedding shape: ", x.shape)
        # print("Rearranged shape: ", x.shape)
        # x = self.graph_conv(x)
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


#### OLD GRAPHFORMER (Graph as first layer)####

# class GraphFormer(nn.Module):
#     def __init__(self, seq_len, K, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5):
#         super(GraphFormer, self).__init__()
#         # self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.spatial_conv = nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1))
#         self.graph_conv = _GraphConvolution(seq_len, seq_len, 22, K)
#         self.spatial_embedding = _SpatialEmbedding(emb_size)
#         self.transformer_encoder = _TransformerEncoder(
#             depth, emb_size, nhead, expansion, dropout)
#         self.clshead = ClassificationHead(194*emb_size, num_classes)

#     def forward(self, x):
#         # x = torch.unsqueeze(x, dim=1)
#         x = x.unsqueeze(dim=1)
#         # print("Embedding shape: ", x.shape)
#         # x = rearrange(x, 'b c t e -> b e c t')
#         # print("Rearranged shape: ", x.shape)
#         x = self.graph_conv(x)
#         # x = self.spatial_conv(x)
#         x = self.spatial_embedding(x)
#         # print("Spatial conv shape: ", x.shape)
#         x = x.squeeze(dim=2)
#         # print("Squeezed shape: ", x.shape)
#         x = rearrange(x, 'b e t -> b t e')
#         x = self.transformer_encoder(x)
#         # print("Transformer shape: ", x.shape)
#         # x, out = self.clshead(x)
#         out = self.clshead(x)
#         return out


#### SIMPLE GRAPHFORMER ####
# from layers import _GraphConvolution, _ResidualAdd, _SpatialEmbedding, _TransformerEncoder, ClassificationHead
# from einops import rearrange
# import torch
# from torch import nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange, Reduce
# import math
# from torch_geometric.nn import GCNConv, SGConv
# from torch_geometric.utils import dense_to_sparse
# from layers import *
