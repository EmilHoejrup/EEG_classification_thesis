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
# from conformer import _MultiHeadAttention, _TransformerEncoderBlock, _PositionwiseFeedforward, _TransformerEncoder, _FullyConnected, _FinalLayer, _ResidualAdd
from models import *
import torch
import torch.nn as nn
from datasets import *
from trainer import *
from EEGTransformer import ConformerCopy, EEGTransformer, EEGTransformerEmb
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
from torchviz import make_dot
# from conformer import *
from einops.layers.torch import Rearrange, Reduce
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


# %%
window_size = 8
stride = 11
train_dataset = BNCI_LEFT_RIGHT(
    train=True, window_size=window_size, stride=stride, strategy='permute')
val_dataset = BNCI_LEFT_RIGHT(
    train=False, window_size=window_size, stride=stride, strategy='permute')
images, channels, timepoints = train_dataset.get_X_shape()
vocab_size = train_dataset.get_vocab_size()
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
# model = EEGConformer(n_classes=2, n_channels=22,
#                      input_window_samples=timepoints, add_log_softmax=False, final_fc_length='auto')
# model = ConTransformer(seq_len=timepoints,
#                        n_layers=1, n_head=2, d_model=64, max_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, num_classes=2, )
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=timepoints, final_conv_length='auto')
model = EEGTransformer(seq_len=timepoints, vocab_size=vocab_size, nhead=2, num_classes=2,
                       depth=2, emb_size=22, expansion=4, dropout=0.1)
# model = Transformer(seq_len=timepoints, max_len=timepoints)
# IMPLEMENT weight decay
# %%
images, channels, timepoints = train_dataset.get_X_shape()
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=50, print_metrics=True)

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
# model = ConTransformer(drop_prob=0.1, seq_len=timepoints,
#                        n_layers=1, n_head=2, d_model=32, max_len=timepoints)
# model = Experiment(n_channels=22, seq_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, depth=3, heads=4)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2, add_log_softmax=False,
#                         input_window_samples=timepoints, final_conv_length='auto')
# model = EEGConformer(n_classes=2, n_channels=22,
#                      input_window_samples=timepoints, final_fc_length='auto', add_log_softmax=False)
model = ConformerCopy(seq_len=timepoints, vocab_size=timepoints,
                      nhead=2, num_classes=2, depth=2, emb_size=20, expansion=4, dropout=0.5)
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=50, final_conv_length='auto')
# model = SimpleViT()
# model = EEGTransformer(seq_len=timepoints, nhead=2, num_classes=2,
#                        depth=2, emb_size=22, expansion=4, dropout=0.5)
# model = ATCNet(n_channels=22, n_outputs=2, n_windows=1,
#                add_log_softmax=False, conv_block_kernel_length_1=4, conv_block_kernel_length_2=4, conv_block_pool_size_1=8, conv_block_pool_size_2=7, tcn_kernel_size=2)
# model = ATCNet(in_channels=22, num_classes=2)
# model = TinyVGG(22,100,2, 3)
# model = ViT(num_classes=2, head_channels=22)
# model = EEGNet(chunk_size=timepoints, num_electrodes=22,
#                kernel_1=128, kernel_2=64)
# model = Transformer(seq_len=timepoints, max_len=timepoints, drop_prob=0.5)
# %%

loss_fun = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0625 * 0.01)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)

trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=100, print_metrics=True)
trainer.plot_train_val_scores()

# %%
window_size = 7
stride = 7
train_dataset = BNCI_LEFT_RIGHT_COMPRESSED(
    train=True, window_size=window_size, stride=stride, strategy='compress')
val_dataset = BNCI_LEFT_RIGHT_COMPRESSED(
    train=False, window_size=window_size, stride=stride, strategy='compress')
images, timepoints = train_dataset.get_X_shape()
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
# model = EEGConformer(n_classes=2, n_channels=22,
#  input_window_samples=timepoints, add_log_softmax=False, final_fc_length='auto')
# model = ConTransformer(seq_len=timepoints,
#                        n_layers=1, n_head=2, d_model=64, max_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, num_classes=2, )
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=timepoints, final_conv_length='auto')
model = EEGTransformer(seq_len=timepoints, nhead=2, num_classes=2,
                       depth=2, emb_size=22, expansion=4, dropout=0.1)
model = EEGTransformerEmb(seq_len=timepoints, nhead=2,
                          num_classes=2, dropout=0.5)
# model = Transformer(seq_len=timepoints, max_len=timepoints)
# IMPLEMENT weight decay
# %%
images, timepoints = train_dataset.get_X_shape()
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=50, print_metrics=True)

trainer.plot_train_val_scores()
# %%

window_size = 3
stride = 3
train_dataset = BNCI_LEFT_RIGHT_NEW_PE(
    train=True, window_size=window_size, stride=stride, strategy='permute')
val_dataset = BNCI_LEFT_RIGHT_NEW_PE(
    train=False, window_size=window_size, stride=stride, strategy='permute')
images, channels, timepoints = train_dataset.get_X_shape()
vocab_size = train_dataset.get_vocab_size()
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
# model = EEGConformer(n_classes=2, n_channels=22,
#  input_window_samples=timepoints, add_log_softmax=False, final_fc_length='auto')
# model = ConTransformer(seq_len=timepoints,
#                        n_layers=1, n_head=2, d_model=64, max_len=timepoints)
# model = VanillaTransformer(num_electrodes=22, num_classes=2, )
# model = ShallowFBCSPNet(n_chans=22, n_classes=2,
#                         input_window_samples=timepoints, final_conv_length='auto')
# model = EEGTransformer(seq_len=timepoints, vocab_size=vocab_size, nhead=2, num_classes=2,
#                        depth=2, emb_size=22, expansion=4, dropout=0.1)
model = EEGTransformerEmb(seq_len=timepoints, vocab_size=vocab_size, nhead=2,
                          num_classes=2,  depth=2, emb_size=40, expansion=4, dropout=0.1)
# model = Transformer(seq_len=timepoints, max_len=timepoints)
# IMPLEMENT weight decay
# %%
images, channels, timepoints = train_dataset.get_X_shape()
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=EPOCHS - 1)
trainer = MultiLabelClassifierTrainer(model, train_loader=train_dataloader, scheduler=scheduler,
                                      val_loader=val_dataloader, loss_fun=loss_fun, optimizer=optimizer, device=device)
trainer.fit(epochs=50, print_metrics=True)

trainer.plot_train_val_scores()

# %%

x, y = next(iter(train_dataloader))
x.shape
# %%
z = model(x)
make_dot(z.mean(), params=dict(model.named_parameters()),
         show_attrs=True, show_saved=True).render('attached', format='png')

# %%


def register_hook(model, submodule_name, hook_fn):
    submodule = dict(model.named_modules())[submodule_name]
    return submodule.register_forward_hook(hook_fn)


def hook_fn(module, input, output):
    global hook_output
    hook_output = output.clone()
    hook_output = hook_output.detach().numpy()


def plot_output(output):
    print(output.shape)
    # plt.plot(output[0, 0, :])
    # plt.imshow(output[0, 0, :].reshape(1, -1), cmap='hot', interpolation='nearest')
    plt.imshow(output[5, :, :])
    # plt.plot(output[0, 0, :])


def plot_hooks():
    x, y = next(iter(train_dataloader))
    hook = register_hook(model, 'patch_embedding', hook_fn)
    with torch.no_grad():
        output = model(x)
    plot_output(hook_output)
    hook.remove()


plot_hooks()

# %%
encoder = nn.TransformerEncoderLayer(d_model=72, nhead=2)
x, y = next(iter(train_dataloader))
z = encoder(x)
z.shape

# %%


# %%
