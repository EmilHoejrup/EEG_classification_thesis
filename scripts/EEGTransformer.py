
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class EEGTransformer(nn.Module):
    def __init__(self, seq_len, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(EEGTransformer, self).__init__()
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)

    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.transformer_encoder(x)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out


class EEGTransformerEmb(nn.Module):
    def __init__(self, seq_len, nhead=2, num_classes=2, depth=2, emb_size=22, expansion=4, dropout=0.5):
        super(EEGTransformerEmb, self).__init__()
        self.embedding = nn.Embedding(seq_len, emb_size)
        self.transformer_encoder = _TransformerEncoder(
            depth, emb_size, nhead, expansion, dropout)
        self.clshead = ClassificationHead(seq_len*emb_size, num_classes)

    def forward(self, x):
        # x = rearrange(x, 'b c t -> b t c')
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # x, out = self.clshead(x)
        out = self.clshead(x)
        return out


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
