from init import *
from common import *

# [
#   [
#     1499040000000,      // 开盘时间
#     "0.01634790",       // 开盘价
#     "0.80000000",       // 最高价
#     "0.01575800",       // 最低价
#     "0.01577100",       // 收盘价(当前K线未结束的即为最新价)
#     "148976.11427815",  // 成交量
#     1499644799999,      // 收盘时间
#     "2434.19055334",    // 成交额
#     308,                // 成交笔数
#     "1756.87402397",    // 主动买入成交量
#     "28.46694368",      // 主动买入成交额
#     "17928899.62484339" // 请忽略该参数
#   ]
# ]
    
if __name__ == "__main__":
    csv_files_list = glob.glob(ori_data_path + trading_pairs + "/" + trading_interval + "_extracted/" + "*.csv")
    df_queue = load_data(csv_files=csv_files_list)
    with open(pkl_path + "/" + pkl_name, 'wb') as f:
        dill.dump(df_queue, f)
    print("dump data to pkl file successfully!")
    print("data_queue size:", df_queue.qsize())