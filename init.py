import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

ori_data_src = r"./download/spot/daily/klines/"
trading_pair = "BTCUSDT"
trading_interval = "5m"

seq_length = 5 # windows size
lr = 0.001
weight_decay = 0.0001
batch_size = 64
epochs = 100
num_workers = 4

test_interval = 50
save_interval = 10

output_dimension = 4
input_dimension = 11 # number of features
pkl = True # save data as pkl
buffer_size = 1000 # size of the buffer for data loading
d_model = 512 # dimension of the hidden state in the transformer model
nhead = 8 # number of heads in the multi-head attention

