import argparse

from init import *
from common import *
from model import *

parser = argparse.ArgumentParser(description="Train a Transformer model for time series prediction")
parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm", "mlp"], help="Model type to use")
parser.add_argument("--predict_days", type=int, default=0, help="Number of days to predict")
args = parser.parse_args()
assert args.model in ["transformer", "lstm", "mlp"], "Invalid model type specified"

criterion = nn.MSELoss()
match args.model:
    case "transformer":
        model = TransformerModel(input_dim=INPUT_DIMENSION, d_model=D_MODEL, nhead=NHEAD, num_layers=6, 
                                 dim_feedforward=2048, output_dim=OUTPUT_DIMENSION, max_len=SEQ_LEN, mode=0)
        save_path = transformer_path
        criterion = nn.MSELoss()
    case "lstm":
        model = LSTM(dimension=INPUT_DIMENSION)
        save_path = lstm_path
        criterion = nn.MSELoss()
    case "cnnlstm":
        assert abs(int(args.predict_days)) > 0, "predict_days must be greater than 0 for CNN-LSTM"
        model = CNNLSTM(dimension=INPUT_DIMENSION, num_classes=OUTPUT_DIMENSION, predict_days=int(args.predict_days))
        save_path = cnnlstm_path
        criterion = nn.MSELoss()
model = model.to(device, non_blocking=True)

def train(epoch, dataloader, scaler, data_queue=None):
    break

if __name__ == "__main__":
    if torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        print("Using CPU for training")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = CustomSchedule(d_model=D_MODEL, optimizer=optimizer, warmup_steps=WARMUP_STEPS)
    if int(args.predict_days) > 0:
        if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Optimizer.pkl"):
            print("Load model and optimizer from file")
            model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Model.pkl"))
            optimizer.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(args.predict_days) + "_Optimizer.pkl"))
        else:
            print("No model and optimizer file, train from scratch")
    else:
        if os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl") and os.path.exists(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"):
            print("Load model and optimizer from file")
            model.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl"))
            optimizer.load_state_dict(torch.load(save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl"))
        else:
            print("No model and optimizer file, train from scratch")
    
    _datas = []
    with open(train_pkl_path, 'rb') as f:
        _data_queue = dill.load(f)
        while _data_queue.empty() is False:
            try:
                _datas.append(_data_queue.get())
            except queue.Empty:
                break
            init_bar = tqdm(total=len(_datas), desc="Loading data")
            for _data in _datas:
                init_bar.update(1)
                _data = _data.fillna(_data.median(numeric_only=True))
                if _data.empty:
                    continue
                data_queue.put(_data)
            init_bar.close()
    print("Data loaded successfully, size:", data_queue.qsize())

    