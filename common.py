import targets

import mplfinance as mpf
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from init import *
from prefetch_generator import BackgroundGenerator
from cycler import cycler

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2).item()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

def compare_tensor(original, targets):
    assert original.size() == targets.size()
    result = torch.zeros(original.size()[0])
    for index in range(original.size()[0]):
        if original[index] == targets[index]:
            result[index] = 0
        elif original[index] > targets[index]:
            result[index] = 1
        else:
            result[index] = -1
    return result

class Crypto_Data(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, transform=None, dataFrame=None, label_num=1, predict_days=0, trend=0):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.predict_days = predict_days
            self.data = self.load_data(dataFrame)
            self.normalize_data()
            self.value, self.label = self.generate_value_label_tensors(label_num)
            self.trend=trend
        except Exception as e:
            print(e)
            return None

    def load_data(self, dataFrame):
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            path = train_data_path
        else:
            path = test_data_path

        if dataFrame is None:
            with open(path) as f:
                data = np.loadtxt(f, delimiter=",")
        else:
            data = dataFrame.values

        return data[:, 0:INPUT_DIMENSION]

    def normalize_data(self):
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            for i in range(len(self.data[0])):
                if self.mode in [0, 2]:
                    mean_list.append(np.mean(self.data[:, i]))
                    std_list.append(np.std(self.data[:, i]))

                self.data[:, i] = (self.data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
        else:
            test_mean_list.clear()
            test_std_list.clear()
            for i in range(len(self.data[0])):
                if self.mode not in [0, 2]:
                    test_mean_list.append(np.mean(self.data[:, i]))
                    test_std_list.append(np.std(self.data[:, i]))

                self.data[:, i] = (self.data[:, i] - test_mean_list[i]) / (test_std_list[i] + 1e-8)
        return self.data

    def generate_value_label_tensors(self, label_num):
        if self.mode in [0, 1]:
            value = torch.rand(self.data.shape[0] - SEQ_LEN, SEQ_LEN, self.data.shape[1])
            if self.predict_days > 0:
                label = torch.rand(self.data.shape[0] - SEQ_LEN, self.predict_days, label_num)
            elif self.predict_days <= 0:
                label = torch.rand(self.data.shape[0] - SEQ_LEN, label_num)

            for i in range(self.data.shape[0] - SEQ_LEN):
                _value_tmp = np.copy(np.flip(self.data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, self.data.shape[1]), 0))
                value[i, :, :] = torch.from_numpy(_value_tmp)
                _tmp = []
                for index in range(len(use_list)):
                    if use_list[index] == 1:
                        if self.predict_days <= 0:
                            _tmp.append(self.data[i, index])
                        elif self.predict_days > 0:
                            _tmp.append(self.data[i:i+self.predict_days, index])
                if self.predict_days <= 0:
                    label[i, :] = torch.Tensor(np.array(_tmp))
                elif self.predict_days > 0:
                    label[i, :, :] = torch.Tensor(np.array(_tmp)).permute(1,0)
        elif self.mode == 2:
            value = torch.rand(1, SEQ_LEN, self.data.shape[1])
            if self.predict_days <= 0:
                label = torch.rand(1, label_num)
            elif self.predict_days > 0:
                label = torch.rand(1, self.predict_days, label_num)
            _value_tmp = np.copy(np.flip(self.data[0:SEQ_LEN, :].reshape(SEQ_LEN, self.data.shape[1]), 0))
            value[0, :, :] = torch.from_numpy(_value_tmp)
            if self.trend == 1:
                if self.predict_days > 0:
                    label[i][0] = compare_tensor(label[i][0], value[0][-1][:OUTPUT_DIMENSION])
        
        # if self.trend == 1:
        #     if self.predict_days > 0:
        #         label = compare_tensor(label[0][0], value[0][-1][:OUTPUT_DIMENSION])

        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    def __getitem__(self, index):
        return self.value[index], self.label[index]

    def __len__(self):
        return len(self.value)

def cmp_append(data, cmp_data):  
    # while len(data) < len(cmp_data):
    #     data.append(0)
    if len(cmp_data) - len(data) > 0:
        data += [0] * (len(cmp_data) - len(data))
    # data = np.nan_to_num(data)
    return data

def add_targets(df):
    times = np.array(df["open_time"])
    close = np.array(df["close_price"])
    hpri = np.array(df["high_price"])
    lpri = np.array(df["low_price"])
    vol = np.array(df["trading_volume"])

    macd_dif, macd_dea, macd_bar = targets.MACD(close)
    df["macd_dif"] = cmp_append(macd_dif, df)
    df["macd_dea"] = cmp_append(macd_dea, df)
    df["macd_bar"] = cmp_append(macd_bar, df)
    k, d, j = targets.KDJ(close, hpri, lpri)
    df['k'] = cmp_append(k, df)
    df['d'] = cmp_append(d, df)
    df['j'] = cmp_append(j, df)
    boll_upper, boll_mid, boll_lower = targets.BOLL(close)
    df['boll_upper'] = cmp_append(boll_upper, df)
    df['boll_mid'] = cmp_append(boll_mid, df)
    df['boll_lower'] = cmp_append(boll_lower, df)
    cci = targets.CCI(close, hpri, lpri)
    df['cci'] = cmp_append(cci, df)
    pdi, mdi, adx, adxr = targets.DMI(close, hpri, lpri)
    df['pdi'] = cmp_append(pdi, df)
    df['mdi'] = cmp_append(mdi, df)
    df['adx'] = cmp_append(adx, df)
    df['adxr'] = cmp_append(adxr, df)
    taq_up, taq_mid, taq_down = targets.TAQ(hpri, lpri, 5)
    df['taq_up'] = cmp_append(taq_up, df)
    df['taq_mid'] = cmp_append(taq_mid, df)
    df['taq_down'] = cmp_append(taq_down, df)
    trix, trma = targets.TRIX(close)
    df['trix'] = cmp_append(trix, df)
    df['trma'] = cmp_append(trma, df)
    atr = targets.ATR(close, hpri, lpri)
    df['atr'] = cmp_append(atr, df)

    df_queue.put(df)
    return df_queue

def load_data(csv_files=None):
    dataframes = []
    for csv_file in tqdm(csv_files, desc="Loading data"):
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, header=None, names=name_list)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    if "ignore_data" in combined_df.columns:
        combined_df = combined_df.drop(columns=["ignore_data"])
    combined_df = add_targets(combined_df)
    return combined_df

def generate_attention_mask(input_data):
    mask = (input_data != -0.0).any(dim=-1)  # Find non-padding positions 
    mask = mask.to(torch.float32)
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.T
    return mask

def data_wash(dataset,keepTime=False):
    if keepTime:
        dataset.fillna(axis=1,method='ffill')
    else:
        dataset.dropna()
    df_queue.put(dataset)
    return dataset
    
def draw_Kline(df, period, symbol):
    kwargs = dict(
        type='candle',
        mav=(7, 30, 60),
        volume=True,
        title=f'\nA_stock {symbol} candle_line',
        ylabel='OHLC Candles',
        ylabel_lower='Shares\nTraded Volume',
        figratio=(15, 10),
        figscale=2)

    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='i',
        wick='i',
        volume='in',
        inherit=True)

    s = mpf.make_mpf_style(
        gridaxis='both',
        gridstyle='-.',
        y_on_right=False,
        marketcolors=mc)

    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['dodgerblue', 'deeppink',
               'navy', 'teal', 'maroon', 'darkorange',
               'indigo'])

    mpl.rcParams['lines.linewidth'] = .5

    mpf.plot(df,
             **kwargs,
             style=s,
             show_nontrading=False)

    mpf.plot(df,
             **kwargs,
             style=s,
             show_nontrading=False,
             savefig=f'A_stock-{symbol} {period}_candle_line.jpg')
    plt.show()

def save_model(model, optimizer, save_path, best_model=False, predict_days=0):
    if predict_days > 0:
        if best_model is False:
            torch.save(model.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model.pkl")
            torch.save(optimizer.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer.pkl")
        elif best_model is True:
            torch.save(model.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Model_best.pkl")
            torch.save(optimizer.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_pre" + str(predict_days) + "_Optimizer_best.pkl")
    else:
        if best_model is False:
            torch.save(model.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model.pkl")
            torch.save(optimizer.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer.pkl")
        elif best_model is True:
            torch.save(model.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Model_best.pkl")
            torch.save(optimizer.state_dict(), save_path + "_out" + str(OUTPUT_DIMENSION) + "_time" + str(SEQ_LEN) + "_Optimizer_best.pkl")

def thread_save_model(model, optimizer, save_path, best_model=False, predict_days=0):
    _model = copy.deepcopy(model)
    _optimizer = copy.deepcopy(optimizer)
    data_thread = threading.Thread(target=save_model, args=(_model, _optimizer, save_path, best_model, predict_days,))
    data_thread.start()

def deep_copy_queue(q):
    new_q = multiprocessing.Queue()
    # new_q = queue.Queue()
    temp_q = []
    while not q.empty():
        try:
            item = q.get_nowait()
            temp_q.append(item)
        except queue.Empty:
            break
    for item in temp_q:
        new_q.put(item)
        q.put(item)
    return new_q

