from init import *
from common import *

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
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

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

class Crypto_queue_dataset(Dataset):
    # mode 0:train 1:test 2:predict
    def __init__(self, mode=0, data_queue=None, label_num=1, buffer_size=100, total_length=0, predict_days=0, trend=0):
        try:
            assert mode in [0, 1, 2]
            self.mode = mode
            self.data_queue = data_queue
            self.label_num = label_num
            self.buffer_size = buffer_size
            self.buffer_index = 0
            self.value_buffer = []
            self.label_buffer = []
            self.total_length = total_length
            self.predict_days = predict_days
            self.trend=trend
        except Exception as e:
            print(e)
            return None

    def load_data(self):
        if self.data_queue.empty():
            return None
        else:
            try:
                dataFrame = self.data_queue.get(timeout=30)
            except queue.Empty:
                return None
            dataFrame.drop(['open_time', 'close_time'], axis=1, inplace=True)
            # dataFrame = dataFrame.dropna()
            # dataFrame = dataFrame.fillna(-0.0)
            dataFrame = dataFrame.fillna(dataFrame.median(numeric_only=True))
            data = dataFrame.values[:, 0:INPUT_DIMENSION]
            return data

    def normalize_data(self, data):
        if self.mode in [0, 2]:
            mean_list.clear()
            std_list.clear()
            for i in range(len(data[0])):
                if self.mode in [0, 2]:
                    mean_list.append(np.mean(data[:, i]))
                    std_list.append(np.std(data[:, i]))

                data[:, i] = (data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
        else:
            test_mean_list.clear()
            test_std_list.clear()
            for i in range(len(data[0])):
                if self.mode not in [0, 2]:
                    test_mean_list.append(np.mean(data[:, i]))
                    test_std_list.append(np.std(data[:, i]))

                data[:, i] = (data[:, i] - test_mean_list[i]) / (test_std_list[i] + 1e-8)
        return data

    def generate_value_label_tensors(self, data, label_num):
        value = torch.rand(data.shape[0] - SEQ_LEN, SEQ_LEN, data.shape[1])
        if self.predict_days > 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, self.predict_days, label_num)
        elif self.predict_days <= 0:
            label = torch.rand(data.shape[0] - SEQ_LEN, label_num)

        for i in range(data.shape[0] - SEQ_LEN):
            _value_tmp = np.copy(np.flip(data[i + 1:i + SEQ_LEN + 1, :].reshape(SEQ_LEN, data.shape[1]), 0))
            value[i, :, :] = torch.from_numpy(_value_tmp)

            _tmp = []
            for index in range(len(use_list)):
                if use_list[index] == 1:
                    if self.predict_days > 0:
                        _tmp.append(data[i:i+self.predict_days, index])
                    elif self.predict_days <= 0:
                        _tmp.append(data[i, index])
            if self.predict_days > 0:
                label[i, :, :] = torch.Tensor(np.array(_tmp)).permute(1,0)
            elif self.predict_days <= 0:
                label[i, :] = torch.Tensor(np.array(_tmp))
            if self.trend == 1:
                if self.predict_days > 0:
                    label[i][0] = compare_tensor(label[i][0], value[0][-1][:OUTPUT_DIMENSION])
        _value = value.flip(0)
        _label = label.flip(0)
        return _value, _label

    def process_data(self):
        # Check if there is data in the queue
        if self.data_queue.empty():
            return None

        for _ in range(self.buffer_size):  # Loop for buffer_size times
            try:
                raw_data = self.load_data()
                if raw_data is not None:
                    while len(raw_data) < SEQ_LEN:
                        raw_data = self.load_data()
                        if raw_data is None:
                            break
                    if raw_data is not None:
                        normalized_data = self.normalize_data(raw_data)
                        value, label = self.generate_value_label_tensors(normalized_data, self.label_num)
                        
                        self.value_buffer.extend(value)
                        self.label_buffer.extend(label)
                    else:
                        break
                else:
                    break
            except:
                continue

        if len(self.value_buffer) == 0 or len(self.label_buffer) == 0:
            return None

    def __getitem__(self, index):
        try:
            while self.buffer_index >= len(self.value_buffer):
                self.value_buffer.clear()
                self.label_buffer.clear()
                self.buffer_index = 0
                ans = self.process_data()
                if ans is None:
                    break
            if self.buffer_index >= len(self.value_buffer):
                return None, None
            value, label = self.value_buffer[self.buffer_index], self.label_buffer[self.buffer_index]
            self.buffer_index += 1
            return value, label
        except Exception as e:
            print(e)
            return None, None

    def __len__(self):
        if self.data_queue is None:
            return len(self.value_buffer)
        else:
            return self.total_length
        # return self.total_length

class TransformerEncoderLayerWithNorm(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm

class TransformerDecoderLayerWithNorm(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        if norm is not None:
            self.norm1 = norm
            self.norm2 = norm
            self.norm3 = norm

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, max_len=5000, mode=0):
        super(TransformerModel, self).__init__()

        # self.embedding = nn.Linear(input_dim, d_model)
        assert d_model % 2 == 0, "d_model must be a multiple of 2"
        self.embedding = MLP(input_dim, d_model//2, d_model)  # Replace this line
        self.positional_encoding = None

        if mode in [0]:
            dropout = 0.5
        elif mode in [1 ,2]:
            dropout = 0

        self.transformer_encoder_layer = TransformerEncoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model), dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)

        self.transformer_decoder_layer = TransformerDecoderLayerWithNorm(d_model, nhead, dim_feedforward, norm=nn.LayerNorm(d_model), dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers)

        self.target_embedding = nn.Linear(output_dim, d_model)
        self.pooling = nn.AdaptiveAvgPool1d
        self.fc = nn.Linear(d_model, output_dim)

        self.d_model = d_model
        self.output_dim = output_dim

        self._initialize_weights()

    def forward(self, src, tgt, predict_days=0):
        src = src.permute(1, 0, 2) # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)

        attention_mask = generate_attention_mask(src)
        src_embedding = self.embedding(src)
        src_seq_length = src.size(0)
        src_batch_size = src.size(1)

        if self.positional_encoding is None or self.positional_encoding.size(0) < src_seq_length:
            self.positional_encoding = self.generate_positional_encoding(src_seq_length, self.d_model).to(src.device)

        src_positions = torch.arange(src_seq_length, device=src.device).unsqueeze(1).expand(src_seq_length, src_batch_size)
        src = src_embedding + self.positional_encoding[src_positions]

        memory = self.transformer_encoder(src, src_key_padding_mask=attention_mask)

        if predict_days <= 0:
            tgt = tgt.unsqueeze(0)
        else:
            tgt = tgt.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        tgt_embedding = self.target_embedding(tgt)
        tgt_seq_length = tgt.size(0)

        tgt_positions = torch.arange(tgt_seq_length, device=tgt.device).unsqueeze(1).expand(tgt_seq_length, src_batch_size)
        tgt = tgt_embedding + self.positional_encoding[tgt_positions]

        output = self.transformer_decoder(tgt, memory)

        if predict_days <= 0:
            output = output.permute(1, 2, 0)  # (seq_len, batch_size, d_model) -> (batch_size, d_model, seq_len)
            pooled_output = self.pooling(1)(output) # (batch_size, d_model, seq_len) -> (batch_size, d_model, seq_len)
            output = self.fc(pooled_output.squeeze(2))
        else:
            output = output.permute(1, 2, 0)  # (seq_len, batch_size, d_model) -> (batch_size, d_model, seq_len)
            pooled_output = self.pooling(predict_days)(output)
            output = self.fc(pooled_output.permute(0,2,1))
            
        return output

    def generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(1)
        return pe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LSTM(nn.Module):
    def __init__(self,dimension):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=dimension,hidden_size=128,num_layers=3,batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(in_features=128,out_features=16)
        self.linear2 = nn.Linear(16,OUTPUT_DIMENSION)
        self.LeakyReLU = nn.LeakyReLU()
        # self.ELU = nn.ELU()
        # self.ReLU = nn.ReLU()
    def forward(self,x, tgt=None, predict_days=0):
        # self.lstm.flatten_parameters()
        # lengths = [s.size(0) for s in x] # 获取数据真实的长度
        # x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # out_packed, _ = self.lstm(x_packed)
        # out, lengths = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        out,_=self.lstm(x)
        x = out[:,-1,:]        
        x = self.linear1(x)
        x = self.LeakyReLU(x)
        # x=self.ELU(x)
        x = self.linear2(x)
        if predict_days > 0:
            x = x.unsqueeze(1)
        return x
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        attn_weights = self.softmax(self.attn(outputs))
        return torch.bmm(attn_weights.transpose(1, 2), outputs).squeeze(1)
    
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes=2, predict_days=1, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3, batch_first=True)
        self.attention = Attention(256)
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)  # new Dropout layer after fc1
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)  # new Dropout layer after fc2
        self.predict_days = predict_days
       
    def forward(self, x_3d, _tgt, _predict_days=1):
        self.lstm.flatten_parameters()
        batch_size, seq_len, C = x_3d.size() # batch_size, seq_len, input_dim
        tar = torch.zeros(batch_size, 1, OUTPUT_DIMENSION).to(x_3d.device)
        for index in range(x_3d.size(0)):
            tar[index] = x_3d[index][-1][:OUTPUT_DIMENSION]
        x_3d = x_3d.transpose(1, 2)  # change the shape to (batch_size, in_channels, seq_len)
        c_out = F.relu(self.conv1(x_3d))
        c_out = F.relu(self.conv2(c_out))
        r_in = c_out.transpose(1, 2)  # change the shape back to (batch_size, seq_len, -1)
        r_out, _ = self.lstm(r_in)
        attn_out = self.attention(r_out) # batch_size, hidden_size
        x = F.relu(self.fc1(attn_out))
        x = self.dropout1(x)  # apply Dropout after fc1
        x = self.fc2(x)
        x = self.dropout2(x)  # apply Dropout after fc2
        return x.view(batch_size, self.predict_days, -1)