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


def extract_label(stock_name, market_name, label_date, data_path):
    if stock_name not in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
        kline_data = pd.read_csv(f'{data_path}/kline_day/{market_name}/{stock_name}.csv', index_col='date')
        # label
        try:
            label = kline_data.at[label_date, 'ret']
            if np.isnan(label) or np.isinf(label):
                label = 0.0
        except KeyError as e:
            label = 0.0
        return label
    else:
        return 0.0


def one_day(date, market_name, data_path, save_path, stock_list, date_array, 
            horizon = 1, num_worker=20):
    label_date = date_array[np.where(date_array == date)[0].item() + horizon]
    
    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_label)
                                               (stock_name, market_name, label_date, data_path)
                                               for stock_name in stock_list)

    label_tensor = np.zeros([stock_list.shape[0], 1])

    for i in range(len(one_day_list)):  # type: ignore
        label_tensor[i, 0] = one_day_list[i]

    np.save(f'{save_path}/{date}/label.npy', label_tensor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='NASDAQ',
                        help='The market name (NASDAQ or NYSE)')
    parser.add_argument(
        '--start-time',
        type=str,
        default='2013-01-02',
        help='Dataset start time'
    )
    parser.add_argument(
        '--end-time',
        type=str,
        default='2017-12-08',
        help='Dataset end time'
    )
    parser.add_argument('--worker', type=int, default=75,
                        help='The number of process.')
    args = parser.parse_args()

    data_path = '/home/zfzhu/Documents/GRACE_data'
    market_name = args.market
    start_time = args.start_time
    end_time = args.end_time
    lag_order = 16
    horizon = 1
    P = 10
    skip_exsting = True
    save_path = f'{data_path}/tensors/{market_name}'

    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    tickers = np.genfromtxt(os.path.join(data_path, tickers_fname), dtype=str, delimiter='\t', skip_header=False)
    tickers = np.hstack([tickers, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])

    FF5_df = pd.read_csv(f'{data_path}/FF5.csv', index_col = 'date')
    date_array = (FF5_df.index.values)

    selected_date_array = (FF5_df.loc[start_time:end_time, :].index.values)

    for date in tqdm(selected_date_array, desc='Label construction'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/label.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
            os.makedirs(date_save_path)

        one_day(date, market_name, data_path, save_path, tickers, date_array, horizon, num_worker=args.worker)