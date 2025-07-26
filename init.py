import torch
import queue
import os
import glob
import dill
import copy
import threading
# import multiprocessing
import re

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = False
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    pin_memory = True

SEQ_LEN = 5 # windows size
LR = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
EPOCHS = 100
NUM_WORKERS = 0 # number of workers for data loading, set to 0 for no multiprocessing

TEST_INTERVAL = 50
SAVE_INTERVAL = 100

OUTPUT_DIMENSION = 4
INPUT_DIMENSION = 29 # number of features
PKL = True # save data as pkl
BUFFER_SIZE = 1000 # size of the buffer for data loading
D_MODEL = 512 # dimension of the model
NHEAD = 8 # number of heads in the multi-head attention
WARMUP_STEPS = 60000 # number of warmup steps for the learning rate scheduler

loss_list=[]
data_list=[]
mean_list=[]
std_list=[]
test_mean_list = []
test_std_list = []
safe_save = False
# data_queue=multiprocessing.Queue()
data_queue=queue.Queue()
test_queue=queue.Queue()
crypto_data_queue=queue.Queue()
crypto_list_queue = queue.Queue()
csv_queue=queue.Queue()
df_queue=queue.Queue()

name_list = [
    "open_time", "open_price", "high_price", "low_price", "close_price",
    "trading_volume", "close_time", "transaction_volume", "number_of_trades",
    "active_buy_volume", "active_sell_volume", "ignore_data"
]
    # "open_price", "high_price", "low_price", "close_price",
    # "trading_volume", "transaction_volume", "number_of_trades",
    # "active_buy_volume", "active_sell_volume", "ignore_data"
use_list = [
    1,1,1,1,
    1,1,1,
    1,1,1
]
show_list = [
    0,1,1,1,1,
    0,0,0,0,
    0,0,0
]
OUTPUT_DIMENSION = sum(use_list)
assert OUTPUT_DIMENSION > 0
last_buffer_loop_count = 0
last_value_buffer = []
last_label_buffer = []

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

ori_data_path = r"downloads/spot/daily/klines/"
trading_pairs = "BTCUSDT"
trading_interval = "5m"
handle_path = r"data_handle/"
train_data_path = os.path.join(handle_path, "train_data.csv")
test_data_path = os.path.join(handle_path, "test_data.csv")
pkl_path =  r"pkl_handle/"
pkl_name = "data.pkl"
train_pkl_path = pkl_path + pkl_name
lstm_path = r"model/lstm_model/"
transformer_path = r"model/transformer_model/"
cnnlstm_path = r"model/cnnlstm_model/"

check_path(handle_path)
check_path(pkl_path)
check_path(lstm_path)
check_path(transformer_path)
check_path(cnnlstm_path)