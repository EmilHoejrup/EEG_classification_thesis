
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, SGConv, ChebConv, GraphConv
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import math


class _SpatialEmbedding(nn.Module):
    """
    A class representing the spatial embedding layer.

    Args:
        emb_size (int): The size of the embedding.

    Attributes:
        spatial (nn.Sequential): The sequential module representing the spatial embedding layer.

    """

    def __init__(self, emb_size):
        super(_SpatialEmbedding, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, emb_size), (1, 1)),
            nn.Conv2d(emb_size, emb_size, (22, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d((1, 15), (1, 5))
        )

    def forward(self, x):
        """
        Forward pass of the spatial embedding layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the spatial embedding layer.

        """
        return self.spatial(x)


class _PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to input tensors.

    Args:
        emb_size (int): The size of the input embedding.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.
    """

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
        """
        Forward pass of the positional encoding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class _MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_model (int): The dimensionality of the input and output features.
        nhead (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Default: 0.1.

    Attributes:
        d_model (int): The dimensionality of the input and output features.
        nhead (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        scale (float): The scaling factor for the attention scores.
        qkv (nn.Linear): Linear layer for computing the query, key, and value projections.
        fc (nn.Linear): Linear layer for the final output projection.
        dropout (nn.Dropout): Dropout layer for regularization.

    """

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
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length, sequence_length).
                Each element should be True for positions that should be masked and False for positions that should be attended to.
                Default: None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, d_model).

        """
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
    """
    A module that performs residual addition.

    Args:
        fn (nn.Module): The module to be applied.

    Attributes:
        fn (nn.Module): The module to be applied.

    Methods:
        forward(x, **kwargs): Performs the forward pass of the module.

    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Performs the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor.

        """
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _TransformerEncoder(nn.Sequential):
    """
    A stack of Transformer Encoder blocks.

    Args:
        depth (int): The number of Transformer Encoder blocks in the stack.
        emb_size (int): The input and output feature dimension.
        nhead (int, optional): The number of attention heads in each block. Default is 2.
        expansion (int, optional): The expansion factor for the feed-forward network in each block. Default is 4.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, depth, emb_size, nhead=2, expansion=4, dropout=0.1):
        super().__init__(*[
            _TransformerEncoderBlock(emb_size, nhead, dropout=dropout)
            for _ in range(depth)
        ])


class _TransformerEncoderBlock(nn.Module):
    """
    This class represents a single block of the Transformer Encoder.

    Args:
        emb_size (int): The input embedding size.
        nhead (int, optional): The number of attention heads. Defaults to 2.
        expansion (int, optional): The expansion factor for the position-wise feedforward network. Defaults to 4.
        dropout (float, optional): The dropout probability. Defaults to 0.5.
    """

    def __init__(self, emb_size, nhead=2, expansion=4, dropout=0.5):
        super().__init__()
        self.attn = _MultiHeadAttention(emb_size, nhead, dropout=dropout)
        self.ff = _PositionwiseFeedforward(
            emb_size, expansion=expansion, dropout=dropout)

    def forward(self, x):
        """
        Perform a forward pass through the Transformer Encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = _ResidualAdd(self.attn)(x)
        x = _ResidualAdd(self.ff)(x)
        return x


class _PositionwiseFeedforward(nn.Module):
    """
    This class implements the position-wise feedforward neural network layer.

    Args:
        emb_size (int): The input embedding size.
        expansion (int): The expansion factor for the hidden layer size.
        dropout (float, optional): The dropout probability. Default is 0.1.

    Attributes:
        net (nn.Sequential): The sequential neural network module.

    """

    def __init__(self, emb_size, expansion, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x):
        """
        Forward pass of the position-wise feedforward neural network layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.net(x)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class GraphConstructor(nn.Module):
    """
    An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_
    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(
            self, nnodes: int, k: int, dim: int, alpha: float):
        super(GraphConstructor, self).__init__()

        self._embedding1 = nn.Embedding(nnodes, dim)
        self._embedding2 = nn.Embedding(nnodes, dim)
        self._linear1 = nn.Linear(dim, dim)
        self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
            self, idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.
        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """
        nodevec1 = self._embedding1(idx)
        nodevec2 = self._embedding2(idx)

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A


class _GraphConvolution(nn.Module):
    """
    Graph Convolution layer implementation.

    Args:
        seq_len (int): Length of the input sequence.
        out_features (int): Number of output features.
        channels (int): Number of channels.
        K (int): Number of graph convolution layers.
        dropout (float, optional): Dropout probability. Default is 0.5.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        channels (int): Number of channels.
        K (int): Number of graph convolution layers.
        graph_constructor (GraphConstructor): Graph constructor object.
        graph_conv (SGConv): Graph convolution layer.
        relu (ReLU): ReLU activation function.
        dropout (float): Dropout probability.

    """

    def __init__(self, seq_len, out_features, channels, K, dropout=0.5):
        super(_GraphConvolution, self).__init__()
        self.in_features = seq_len
        self.out_features = out_features
        self.channels = channels
        self.K = K
        self.graph_constructor = GraphConstructor(
            nnodes=channels, k=K, dim=seq_len, alpha=0.1)
        # self.graph_conv = SGConv(seq_len, out_features, K=K)
        # self.graph_conv = GCNConv(seq_len, out_features)
        self.graph_conv = ChebConv(seq_len, out_features, K=K)
        # self.graph_conv = GraphConv(seq_len, out_features)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass of the graph convolution layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying graph convolution.

        """
        idx = torch.arange(self.channels).long().to(x.device)
        A = self.graph_constructor(idx)
        A, attr = dense_to_sparse(A)
        out = self.graph_conv(x, A, attr)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout)
        return out


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

# class ClassificationHead(nn.Module):
#     """
#     A classification head module that takes an input tensor and produces
#     class predictions.

#     Args:
#         n_classes (int): The number of output classes.

#     Attributes:
#         fc (nn.Sequential): The fully connected layers of the classification head.

#     Methods:
#         forward(x): Performs a forward pass through the network.

#     """

#     def __init__(self, n_classes):
#         super().__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(0, 256),  # Placeholder for dynamic input size
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 32),
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, n_classes)
#         )

#     def forward(self, x):
#         """
#         Performs a forward pass through the network.

#         Args:
#             x (torch.Tensor): The input tensor.

#         Returns:
#             torch.Tensor: The output tensor containing class predictions.

#         """
#         # Compute the input size dynamically
#         input_size = x.size(1) * x.size(2)

#         # Update the first linear layer with the dynamic input size
#         self.fc[0] = nn.Linear(input_size, 256)

#         # Flatten the input
#         x = x.view(x.size(0), -1)

#         # Forward pass through the network
#         out = self.fc(x)
#         return out
