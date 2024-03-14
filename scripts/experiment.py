# %%
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import itertools
import torch.nn.functional as F
import torch.optim as optim
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import Autoencoder
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
from support.utils import *
import yaml
import numpy as np
from support.constants import CONFIG_FILE
from torch.utils.data import DataLoader
from torcheeg.transforms import ToGrid
from torcheeg.models import SimpleViT, ATCNet, VanillaTransformer, EEGNet, ViT
from braindecode.models import ShallowFBCSPNet, EEGConformer
from torch.optim.lr_scheduler import LRScheduler
from new_model import *
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)
# BATCH_SIZE = configs['him_or_her']['batch_size']
BATCH_SIZE = 64

has_gpu = torch.cuda.is_available()
device = 'cpu'
LEARNING_RATE = 0.0001
EPOCHS = 3
# downsampling with moving average
# Hamming window?


class Experiment(nn.Module):
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


# %%
window_size = 3
stride = 3
train_dataset = BNCI_LEFT_RIGHT(
    train=True, window_size=window_size, stride=stride, strategy='permute')
val_dataset = BNCI_LEFT_RIGHT(
    train=False, window_size=window_size, stride=stride, strategy='permute')
images, channels, timepoints = train_dataset.get_X_shape()
# vocab_size = train_dataset.get_vocab_size()
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True, )
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, )
x, y = next(iter(train_dataloader))
x.shape
# %%

# model = XExpTransformer(d_model=200, seq_len=timepoints, ffn_hidden=128,
#                         n_head=4, details=False, drop_prob=0.1, n_layers=1, max_len=vocab_size)
LEARNING_RATE = 0.0001
# model = NewTransformer(embedding_dim=9, vocab_size=vocab_size, dim_ff=32,
#                        seq_len=channels*timepoints, dropout=0.3)
model = Experiment(n_channels=22, seq_len=timepoints, dropout=0.8)
model = EEGConformer(n_classes=2, n_channels=22,
                     input_window_samples=timepoints, add_log_softmax=False, final_fc_length='auto')
# model = ConTransformer(seq_len=timepoints,
#                        n_layers=1, n_head=2, d_model=64, max_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, num_classes=2, )
images, channels, timepoints = train_dataset.get_X_shape()
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=timepoints, final_conv_length='auto')
model = Transformer(seq_len=timepoints, max_len=timepoints)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=100, print_metrics=False)

trainer.plot_train_val_scores()

# %%


#####################

# # model = ShallowFBCSPNet(n_chans=22, n_classes=4,


train_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=True)
val_dataset = BNCI_LEFT_RIGHT_CONTINUOUS(train=False)
train_dataloader = DataLoader(
    dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
# model = SimpleTransformerModelVanilla(
#     input_dim=22, d_model=40, num_layers=4, nhead=2)
# model = Transformer(device='cpu', timepoints=562,
#                     seq_len=562, n_classes=2, details=False)
# model = ATCNet(in_channels=22, num_classes=2,)
# model = EEGConformer(n_outputs=2, n_channels=64, )
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=562, final_conv_length='auto')
b, c, timepoints = train_dataset.get_X_shape()
# model = NewTransformer(seq_len=timepoints*c, vocab_size=(timepoints*c), embedding_dim=9)
model = ConTransformer(drop_prob=0.1, seq_len=timepoints,
                       n_layers=1, n_head=2, d_model=32, max_len=timepoints)
# model = Experiment(n_channels=22, seq_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, depth=3, heads=4)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2, add_log_softmax=False,
#                         input_window_samples=timepoints, final_conv_length='auto')
model = EEGConformer(n_classes=2, n_channels=22,
                     input_window_samples=timepoints, final_fc_length='auto', add_log_softmax=False)
# model = SimpleViT()
# model = TinyVGG(22,100,2, 3)
# model = ViT(num_classes=2, head_channels=22)
# model = EEGNet(chunk_size=timepoints, num_electrodes=22,
#                kernel_1=128, kernel_2=64)
model = Transformer(seq_len=timepoints, max_len=timepoints, drop_prob=0.5)
loss_fun = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=100, print_metrics=False)
trainer.plot_train_val_scores()


# %%
x, y = next(iter(train_dataloader))
x.shape
# %%
rea = Rearrange('b c (w p) -> b (c w) p', p=32)
z = rea(x)
lin = nn.Linear(32, 32)
z = lin(z)
z.shape

# %%
model = VanillaTransformer(num_electrodes=22)
# %%
model(x).shape
# %%
for i in range(6, 10):
    for j in range(2, 5):
        window_size = i
        stride = j

        train_dataset = BNCI_LEFT_RIGHT(
            train=True, window_size=window_size, stride=stride, strategy='permute')
        val_dataset = BNCI_LEFT_RIGHT(
            train=False, window_size=window_size, stride=stride, strategy='permute')
        images, channels, timepoints = train_dataset.get_X_shape()
        # vocab_size = train_dataset.get_vocab_size()
        train_dataloader = DataLoader(
            dataset=train_dataset,  batch_size=BATCH_SIZE, shuffle=True, )
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, )
        x, y = next(iter(train_dataloader))
        x.shape

        # model = XExpTransformer(d_model=200, seq_len=timepoints, ffn_hidden=128,
        #                         n_head=4, details=False, drop_prob=0.1, n_layers=1, max_len=vocab_size)
        LEARNING_RATE = 0.0001
        # model = NewTransformer(embedding_dim=9, vocab_size=vocab_size, dim_ff=32,
        #                        seq_len=channels*timepoints, dropout=0.3)
        # model = Experiment(n_channels=22, seq_len=timepoints, dropout=0.8)
        model = EEGConformer(n_classes=2, n_channels=22,
                             input_window_samples=timepoints, add_log_softmax=False, final_fc_length='auto')
        # model = ConTransformer(seq_len=timepoints,
        #                        n_layers=1, n_head=2, d_model=64, max_len=timepoints)
        # model = VanillaTransformer(num_electrodes=22, num_classes=2, )
        images, channels, timepoints = train_dataset.get_X_shape()
        # model = ShallowFBCSPNet(n_chans=22, n_classes=2,
        #                         input_window_samples=timepoints, final_conv_length='auto')
        loss_fun = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=EPOCHS - 1)
        trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                              val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
        trainer.fit(epochs=200, print_metrics=False)
        print(f"window: {window_size}, stride: {stride}")

        trainer.plot_train_val_scores()

# %%
