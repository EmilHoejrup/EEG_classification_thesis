# %%
from braindecode.models import ATCNet
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
from torcheeg.models import SimpleViT,  VanillaTransformer, EEGNet, ViT
from braindecode.models import ShallowFBCSPNet, EEGConformer, ATCNet
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
model = VanillaTransformer(num_electrodes=22, num_classes=2, )
images, channels, timepoints = train_dataset.get_X_shape()
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=timepoints, final_conv_length='auto')
# model = Transformer(seq_len=timepoints, max_len=timepoints)
# IMPLEMENT weight decay
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=1, print_metrics=False)

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
b, c, timepoints = train_dataset.get_X_shape()
# model = NewTransformer(seq_len=timepoints*c, vocab_size=(timepoints*c), embedding_dim=9)
model = ConTransformer(drop_prob=0.1, seq_len=timepoints,
                       n_layers=1, n_head=2, d_model=32, max_len=timepoints)
# model = Experiment(n_channels=22, seq_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, depth=3, heads=4)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2, add_log_softmax=False,
#                         input_window_samples=timepoints, final_conv_length='auto')
model = EEGConformer(n_classes=2, n_filters_time=40, n_channels=22,
                     input_window_samples=timepoints, final_fc_length='auto', add_log_softmax=False, pool_time_length=25, pool_time_stride=5)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=50, final_conv_length='auto')
# model = SimpleViT()
# model = ATCNet(n_channels=22, n_outputs=2, n_windows=1,
#                add_log_softmax=False, conv_block_kernel_length_1=4, conv_block_kernel_length_2=4, conv_block_pool_size_1=8, conv_block_pool_size_2=7, tcn_kernel_size=2)
# model = ATCNet(in_channels=22, num_classes=2)
# model = TinyVGG(22,100,2, 3)
# model = ViT(num_classes=2, head_channels=22)
# model = EEGNet(chunk_size=timepoints, num_electrodes=22,
#                kernel_1=128, kernel_2=64)
# model = Transformer(seq_len=timepoints, max_len=timepoints, drop_prob=0.5)


loss_fun = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)

trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=50, print_metrics=False)
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

# Define ranges for parameters you want to test
# Example range for input_window_seconds
num_channels = 22
batch_size = 64
num_classes = 2
num_time_samples = 50
input_window_seconds_range = [4.5, 5.0, 5.5, 6.0]
sfreq_range = [100, 200, 250, 500]  # Example range for sfreq

# Generate all combinations of parameters
parameter_combinations = itertools.product(
    input_window_seconds_range, sfreq_range)

# Loop through parameter combinations
for input_window_seconds, sfreq in parameter_combinations:
    try:
        # Initialize model with current parameter combination
        model = ATCNet(n_chans=num_channels, n_outputs=num_classes,
                       input_window_seconds=input_window_seconds, sfreq=sfreq)

        # Construct dummy input
        dummy_input = torch.randn(batch_size, num_channels, num_time_samples)

        # Pass data through the model
        output = model(dummy_input)

        # print(output)
        break

    except RuntimeError as e:
        print(
            f"RuntimeError occurred with parameters: input_window_seconds={input_window_seconds}, sfreq={sfreq}")
        print(e)  # Print the specific error message
        # break  # Terminate the loop

# %%

# Define ranges for parameters you want to test
input_window_seconds_range = [4.5, 5.0, 5.5, 6.0]
sfreq_range = [100, 200, 250, 500]
conv_block_n_filters_range = [16, 32, 64]
conv_block_kernel_length_1_range = [32, 64, 128]
conv_block_kernel_length_2_range = [8, 16, 32]
conv_block_pool_size_1_range = [4, 8, 16]
conv_block_pool_size_2_range = [4, 7, 14]
conv_block_depth_mult_range = [1, 2, 3]
conv_block_dropout_range = [0.1, 0.3, 0.5]
n_windows_range = [3, 5, 7]
att_head_dim_range = [4, 8, 16]
att_num_heads_range = [1, 2, 4]
att_dropout_range = [0.3, 0.5, 0.7]
tcn_depth_range = [1, 2, 3]
tcn_kernel_size_range = [2, 4, 6]
tcn_n_filters_range = [16, 32, 64]
tcn_dropout_range = [0.1, 0.3, 0.5]
concat_range = [True, False]
max_norm_const_range = [0.1, 0.25, 0.5]
add_log_softmax_range = [True, False]

# Generate all combinations of parameters
parameter_combinations = itertools.product(
    input_window_seconds_range, sfreq_range, conv_block_n_filters_range,
    conv_block_kernel_length_1_range, conv_block_kernel_length_2_range,
    conv_block_pool_size_1_range, conv_block_pool_size_2_range,
    conv_block_depth_mult_range, conv_block_dropout_range, n_windows_range,
    att_head_dim_range, att_num_heads_range, att_dropout_range,
    tcn_depth_range, tcn_kernel_size_range, tcn_n_filters_range,
    tcn_dropout_range, concat_range, max_norm_const_range,
    add_log_softmax_range)
# %%
# Loop through parameter combinations
for (input_window_seconds, sfreq, conv_block_n_filters,
     conv_block_kernel_length_1, conv_block_kernel_length_2,
     conv_block_pool_size_1, conv_block_pool_size_2,
     conv_block_depth_mult, conv_block_dropout, n_windows,
     att_head_dim, att_num_heads, att_dropout,
     tcn_depth, tcn_kernel_size, tcn_n_filters,
     tcn_dropout, concat, max_norm_const,
     add_log_softmax) in parameter_combinations:
    try:
        # Initialize model with current parameter combination
        model = ATCNet(n_chans=num_channels, n_outputs=num_classes,
                       input_window_seconds=input_window_seconds, sfreq=sfreq,
                       conv_block_n_filters=conv_block_n_filters,
                       conv_block_kernel_length_1=conv_block_kernel_length_1,
                       conv_block_kernel_length_2=conv_block_kernel_length_2,
                       conv_block_pool_size_1=conv_block_pool_size_1,
                       conv_block_pool_size_2=conv_block_pool_size_2,
                       conv_block_depth_mult=conv_block_depth_mult,
                       conv_block_dropout=conv_block_dropout,
                       n_windows=n_windows, att_head_dim=att_head_dim,
                       att_num_heads=att_num_heads, att_dropout=att_dropout,
                       tcn_depth=tcn_depth, tcn_kernel_size=tcn_kernel_size,
                       tcn_n_filters=tcn_n_filters,
                       tcn_dropout=tcn_dropout, concat=concat,
                       max_norm_const=max_norm_const,
                       add_log_softmax=add_log_softmax)

        # Construct dummy input
        dummy_input = torch.randn(batch_size, num_channels, num_time_samples)

        # Pass data through the model
        output = model(dummy_input)
        print(model)

    except RuntimeError as e:
        print(
            f"RuntimeError occurred with parameters: input_window_seconds={input_window_seconds}, sfreq={sfreq}, "
            f"conv_block_n_filters={conv_block_n_filters}, conv_block_kernel_length_1={conv_block_kernel_length_1}, "
            f"conv_block_kernel_length_2={conv_block_kernel_length_2}, conv_block_pool_size_1={conv_block_pool_size_1}, "
            f"conv_block_pool_size_2={conv_block_pool_size_2}, conv_block_depth_mult={conv_block_depth_mult}, "
            f"conv_block_dropout={conv_block_dropout}, n_windows={n_windows}, att_head_dim={att_head_dim}, "
            f"att_num_heads={att_num_heads}, att_dropout={att_dropout}, tcn_depth={tcn_depth}, "
            f"tcn_kernel_size={tcn_kernel_size}, tcn_n_filters={tcn_n_filters}, tcn_dropout={tcn_dropout}, "
            f"concat={concat}, max_norm_const={max_norm_const}, add_log_softmax={add_log_softmax}")
        print(e)  # Print the specific error message
        # break  # Terminate the loop

# %%

# %%
# Loop through parameter combinations
for input_window_seconds, sfreq in parameter_combinations:
    try:
        # Initialize model with current parameter combination
        model = ATCNet(
            n_chans=num_channels,
            n_outputs=num_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            conv_block_pool_size_1=4,  # Example value, adjust as needed
            conv_block_pool_size_2=3   # Example value, adjust as needed
        )

        # Construct dummy input
        dummy_input = torch.randn(batch_size, num_channels, num_time_samples)

        # Pass data through the model
        output = model(dummy_input)

        # print(output)
        break

    except RuntimeError as e:
        print(
            f"RuntimeError occurred with parameters: input_window_seconds={input_window_seconds}, sfreq={sfreq}")
        print(e)  # Print the specific error message
        # break  # Terminate the loop

# %%
