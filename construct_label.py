import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True


def extract_label(stock_name, buy_date, sell_date, data_path):
    kline_data = pd.read_csv(
        f'{data_path}/kline_day/{stock_name}.csv', index_col='date')
    # label
    try:
        label = (kline_data.at[sell_date, 'Close']/kline_data.at[buy_date, 'Close'])
        if np.isnan(label):
            label = 0.0
    except KeyError as e:
        label = 0.0
    return label


def one_day(date, data_path, save_path, stock_list, date_array, 
            horizon = 1, num_worker=20):
    buy_date = date_array[np.where(date_array == date)[0].item()]
    sell_date = date_array[np.where(date_array == date)[0].item() + horizon]
    
    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_label)
                                               (stock_name, buy_date, sell_date, data_path)
                                               for stock_name in stock_list)

    label_tensor = np.zeros([stock_list.shape[0], 1])

    for i in range(len(one_day_list)):  # type: ignore
        label_tensor[i, 0] = one_day_list[i]

    np.save(f'{save_path}/{date}/label.npy', label_tensor)

parser = argparse.ArgumentParser()
parser.add_argument('--market', type=str, default='NASDAQ',
                    help='The market name (NASDAQ or NYSE)')
parser.add_argument('--worker', type=int, default=1,
                    help='The number of process.')
parser.add_argument('--factor', action='store_true', default=False,
                    help='Whether use factor-augmented graph')
args = parser.parse_args()

data_path = '/data/GRACE_data'
market_name = args.market
start_time = '2013-01-02'
end_time = '2017-12-08'
lag_order = 16
horizon = 1
P = 10
skip_exsting = True
save_path = f'{data_path}/tensors/{market_name}'

tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
tickers = np.genfromtxt(os.path.join(data_path, tickers_fname), dtype=str, delimiter='\t', skip_header=False)
tickers = np.hstack([['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], tickers])

FF5_df = pd.read_csv(f'{data_path}/FF5.csv', index_col = 'date')
date_array = (FF5_df.index.values)

selected_date_array = (FF5_df.loc[start_time:end_time, :].index.values)

for date in tqdm(selected_date_array, desc='construct label'):
    date_save_path = f'{save_path}/{date}'
    if (os.path.exists(f'{date_save_path}/label.npy') & skip_exsting):
        continue
    elif not os.path.exists(date_save_path):
        os.makedirs(date_save_path)

    one_day(date, data_path, save_path, tickers, date_array, horizon, num_worker=args.worker)