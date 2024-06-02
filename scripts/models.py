from layers import _GraphConvolution, _PositionalEncoding, _TransformerEncoder, ClassificationHead, _SpatialEmbedding
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


from layers import _PositionalEncoding, _TransformerEncoder, ClassificationHead


class CollapsedShallowNet(nn.Module):
    """
    A version of the ShallowFBCSPNet model with a combined spatiotemporal convolution instead of separate temporal and spatial convolutions

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input. Defaults to 1001.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        num_kernels (int, optional): Number of kernels in the spatiotemporal convolution. Defaults to 40.
        kernel_size (int, optional): Size of the kernel in the spatiotemporal convolution. Defaults to 25.
        pool_size (int, optional): Size of the pooling window in the spatiotemporal convolution. Defaults to 75.
    """

    def __init__(self, in_channels, num_classes, timepoints=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75):
        super(CollapsedShallowNet, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            in_channels, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size), (1, 15))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2440, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = F.elu(self.spatio_temporal(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ShallowFBCSPNetCopy(nn.Module):
    """An implementation of the ShallowFBCSPNet model from https://arxiv.org/abs/1703.05051 

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input data. Default is 1001.
        dropout (float, optional): Dropout probability. Default is 0.5.
        num_kernels (int, optional): Number of convolutional kernels. Default is 40.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 25.
        pool_size (int, optional): Size of the pooling window. Default is 75.
    """

    def __init__(self, in_channels, num_classes, timepoints=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75):
        super(ShallowFBCSPNetCopy, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (in_channels, 1))
        self.pool = nn.AvgPool2d((1, pool_size), (1, 15))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2440, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CollapsedConformer(nn.Module):
    """
    A version of the Conformer model with a combined spatiotemporal convolution instead of separate temporal and spatial convolutions.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input data. Default is 1000.
        dropout (float, optional): Dropout rate. Default is 0.5.
        num_kernels (int, optional): Number of kernels in the spatiotemporal convolution. Default is 40.
        kernel_size (int, optional): Size of the kernel in the spatiotemporal convolution. Default is 25.
        pool_size (int, optional): Size of the pooling window. Default is 75.
        nhead (int, optional): Number of attention heads in the transformer. Default is 2.
    """

    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75, nhead=2):
        super(CollapsedConformer, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            in_channels, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.projection = nn.Conv2d(num_kernels, num_kernels, (1, 1))
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True, dropout=dropout)
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
        x = torch.unsqueeze(x, dim=2)
        x = self.spatio_temporal(x)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b d t -> b t d')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConformerCopy(nn.Module):
    """ 
    An implementation of the Conformer model from https://ieeexplore.ieee.org/document/9991178.

    This class represents a Conformer model, which is a deep learning model architecture for sequence classification tasks.
    It consists of several convolutional layers, a transformer encoder, and fully connected layers for classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input sequence. Defaults to 1000.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        num_kernels (int, optional): Number of kernels in the convolutional layers. Defaults to 40.
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 25.
        pool_size (int, optional): Size of the pooling window. Defaults to 75.
        nhead (int, optional): Number of attention heads in the transformer encoder. Defaults to 2.
    """

    def __init__(self, in_channels, num_classes, timepoints=1000, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=75, nhead=2):
        super(ConformerCopy, self).__init__()
        maxpool_out = (timepoints - kernel_size + 1) // pool_size

        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (in_channels, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.projection = nn.Conv2d(num_kernels, num_kernels, (1, 1))
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True, dropout=dropout)
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
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b d t -> b t d')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimplePPModel(nn.Module):
    """ 
    A transformer model relying on the EEG data having been discretized using a simplified version of permutation patterns.

    Args:
        vocab_size (int): The size of the vocabulary.
        emb_size (int): The size of the embedding.
        nhead (int, optional): The number of attention heads in the transformer encoder layer. Defaults to 2.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        expansion (int, optional): The expansion factor for the feedforward layer in the transformer encoder layer. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
    """

    def __init__(self, vocab_size, emb_size, nhead=2, num_classes=2, expansion=4, dropout=0.5):
        super(SimplePPModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=expansion*emb_size, dropout=dropout, batch_first=True)
        self.clshead = _ClassificationHead(emb_size, n_classes=num_classes)
        self.positional_encoding = _PositionalEncoding(vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
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
    """ 
    A simple transformer model that takes in raw EEG data as input for classification.

    Args:
        seq_len (int): The length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        nhead (int, optional): The number of attention heads in the transformer encoder. Default is 2.
        num_classes (int, optional): The number of output classes. Default is 2.
        depth (int, optional): The depth of the transformer encoder. Default is 2.
        emb_size (int, optional): The size of the input embeddings. Default is 22.
        expansion (int, optional): The expansion factor in the transformer encoder. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.5.
    """

    def __init__(self, seq_len, vocab_size, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(TransformerOnly, self).__init__()
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)
        self.positional_encoding = _PositionalEncoding(emb_size)

    def forward(self, x):
        x = x.squeeze(-1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.transformer_encoder(x)
        out = self.clshead(x)
        return out


class PPModel(nn.Module):
    """ 
    A transformer model relying on the EEG data having been discretized using permutation patterns.
    
    Args:
        seq_len (int): The length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        nhead (int, optional): The number of attention heads in the transformer encoder. Defaults to 2.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        depth (int, optional): The depth of the transformer encoder. Defaults to 2.
        emb_size (int, optional): The size of the embedding dimension. Defaults to 22.
        expansion (int, optional): The expansion factor in the transformer encoder. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
    """

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
        x = self.transformer_encoder(x)
        out = self.clshead(x)
        return out



class GraphFormer(nn.Module):
    """
    A model that combines multiple graph convolution layers with a transformer model for EEG classification.

    Args:
        seq_len (int): The length of the input sequence.
        K (int): The number of graph nodes.
        nhead (int, optional): The number of attention heads in the transformer encoder. Defaults to 2.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        depth (int, optional): The depth of the transformer encoder. Defaults to 2.
        emb_size (int, optional): The size of the spatial embedding. Defaults to 20.
        expansion (int, optional): The expansion factor in the transformer encoder. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
        avg_pool_kernel (int, optional): The kernel size for average pooling. Defaults to 15.
        avg_pool_stride (int, optional): The stride for average pooling. Defaults to 5.
        num_blocks (int, optional): The number of graph convolution blocks. Defaults to 3.
    """

    def __init__(self, seq_len, K, nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5, avg_pool_kernel=15, avg_pool_stride=5, num_blocks=3):
        super(GraphFormer, self).__init__()
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
        x = self.graph_convolutions(x)
        x = self.avgpool(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b e t -> b t e')
        x = self.transformer_encoder(x)
        out = self.clshead(x)
        return out
