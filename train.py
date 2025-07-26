import argparse

from init import *
from common import *
from model import *

parser = argparse.ArgumentParser(description="Train a Transformer model for time series prediction")
parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm", "mlp"], 
                    help="Model type to use")
parser.add_argument("--predict_days", type=int, default=0, help="Number of days to predict")
parser.add_argument("--trend", type=int, default=0, help="Trend mode: 0 for no trend, 1 for trend") # noqa: E501
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
    global loss_list, safe_save, iteration
    model.train()
    if len(dataloader) > 1:
        subbar = tqdm(total=len(dataloader), leave=False)
    safe_save = False
    for batch in dataloader:
        safe_save = False
        iteration += 1
        data, label = batch
        if batch is None or data is None or label is None:
            if len(dataloader) > 1:
                subbar.update(1)
            continue
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            data = pad_input(data)
            outputs = model.forward(data, label, int(args.predict_days))
            if outputs.shape == label.shape:
                loss = criterion(outputs, label)
            else:
                if len(dataloader) > 1:
                    subbar.update(1)
                continue
        optimizer.zero_grad()
        if is_number(str(loss.item())):
            scaler.scale(loss).backward()
            lr_scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            loss_list.append(loss.item())
        if len(dataloader) > 1:
            subbar.set_description(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item():.6f}")
            subbar.update(1)
        safe_save = True
    if iteration % SAVE_INTERVAL == 0:
        thread_save_model(model, optimizer, save_path, best_model=False, predict_days=int(args.predict_days))
        tqdm.write(f"Model saved at iteration {iteration}")
    if len(dataloader) > 1:
        subbar.close()

if __name__ == "__main__":
    drop_last = False
    last_loss = 1e10
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

    scaler = GradScaler(device=device)
    pbar = tqdm(range(EPOCHS), desc="Training Epochs", leave=False)
    last_eopch = 0
    loss_list = []
    mean_loss = 0
    iteration = 0
    for epoch in range(0, EPOCHS):
        if len(loss_list) == 0:
            mean_loss = 0
        else:
            mean_loss = np.mean(loss_list)
        _data_queue, _size = deep_copy_queue(data_queue)
        assert _size - SEQ_LEN >= 0, "Data queue size must be greater than SEQ_LEN"
        # tqdm.write("epoch: %d, data_queue size after deep copy: %d" % (epoch, data_queue.qsize()))
        # tqdm.write("epoch: %d, _stock_data_queue size: %d" % (epoch, _data_queue.qsize()))

        train_dataset = Crypto_queue_dataset(mode=0, data_queue=_data_queue, label_num=OUTPUT_DIMENSION, 
                                                buffer_size=BUFFER_SIZE, total_length= _size - SEQ_LEN,
                                                predict_days=int(args.predict_days),trend=int(args.trend))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last,
                                        num_workers=NUM_WORKERS, pin_memory=pin_memory, collate_fn=custom_collate)
        pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {mean_loss:.6f}")
        train(epoch+1, train_dataloader, scaler, data_queue=_data_queue)
        pbar.update(1)
pbar.close()