import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
import warnings

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True


def extract_feature(stock_name, market_name, date, previous_date, data_path):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    if stock_name not in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
        kline_data = pd.read_csv(f'{data_path}/kline_day/{market_name}/{stock_name}.csv', index_col='date')
        kline_data['ret_ma5'] = kline_data['ret'].rolling(5).mean()
        kline_data['ret_ma10'] = kline_data['ret'].rolling(10).mean()
        kline_data['ret_ma20'] = kline_data['ret'].rolling(20).mean()
        kline_data['ret_ma30'] = kline_data['ret'].rolling(30).mean()
        
        stock_feature_list = ['ret', 'ret_ma5', 'ret_ma10', 'ret_ma20', 'ret_ma30',
                              'Mkt-RF_loading', 'SMB_loading', 'HML_loading', 'RMW_loading', 'CMA_loading']
        selected_data = pd.DataFrame(columns=stock_feature_list)
        selected_data[stock_feature_list] = kline_data.loc[previous_date:date, stock_feature_list]
        selected_data[stock_feature_list] = (selected_data[stock_feature_list] - kline_data[stock_feature_list].min())/(
            kline_data[stock_feature_list].max() - kline_data[stock_feature_list].min())
    else:
        factor_data = pd.read_csv(f'{data_path}/FF5.csv', index_col='date')
        factor_data['ret'] = factor_data[stock_name]
        factor_data['ret_ma5'] = factor_data['ret'].rolling(5).mean()
        factor_data['ret_ma10'] = factor_data['ret'].rolling(10).mean()
        factor_data['ret_ma20'] = factor_data['ret'].rolling(20).mean()
        factor_data['ret_ma30'] = factor_data['ret'].rolling(30).mean()
        factor_return_list = ['ret', 'ret_ma5', 'ret_ma10', 'ret_ma20', 'ret_ma30']
        factor_loading_list = ['Mkt-RF_loading', 'SMB_loading', 'HML_loading', 'RMW_loading', 'CMA_loading']       
        selected_data = pd.DataFrame(columns=factor_return_list+factor_loading_list)
        selected_data[factor_return_list] = factor_data.loc[previous_date:date, factor_return_list]
        selected_data[factor_return_list] = (selected_data[factor_return_list] - factor_data[factor_return_list].min())/(
            factor_data[factor_return_list].max() - factor_data[factor_return_list].min())
        selected_data[factor_loading_list] = 0.0
        selected_data[f'{stock_name}_loading'] = 1.0
    # remove abnormal value
    selected_data[selected_data.applymap(isnumber)] = np.nan
    selected_data[np.isinf(selected_data)] = np.nan
    # deal with nan
    selected_data.fillna(method='ffill', inplace=True)
    selected_data.fillna(0.0, inplace=True)
    return selected_data


def one_day(date, market_name, data_path, save_path, stock_list, date_array, horizon=1, num_worker=20):
    previous_date = date_array[np.where(date_array == date)[0].item() - lag_order + 1]
    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_feature)
                                               (stock_name, market_name, date, previous_date, data_path)
                                               for stock_name in stock_list)

    feature_tensor = np.zeros([stock_list.shape[0], lag_order, P])

    for i in range(len(one_day_list)):
        if one_day_list[i].values.shape[0] < lag_order:
            tem = one_day_list[i].values.shape[0]
            if tem != 0:
                feature_tensor[i, -tem:, :] = one_day_list[i].values
            else:
                pass
        else:
            feature_tensor[i, :, :] = one_day_list[i].values

    np.save(f'{save_path}/{date}/feature.npy', feature_tensor)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='NASDAQ',
                        help='Market name (NASDAQ or NYSE)')
    parser.add_argument('--start-time', type=str, default='2013-01-02',
                        help='Dataset start time')
    parser.add_argument('--end-time', type=str, default='2017-12-08',
                        help='Dataset end time')
    parser.add_argument('--worker', type=int, default=75,
                        help='Number of processes to use')
    args = parser.parse_args()

    data_path = 'Specify/Your/Data/Path/Here'
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

    for date in tqdm(selected_date_array, desc='Feature construction'):
        date_save_path = f'{save_path}/{date}'
        if (os.path.exists(f'{date_save_path}/feature.npy') & skip_exsting):
            continue
        elif not os.path.exists(date_save_path):
            os.makedirs(date_save_path)

        one_day(date, market_name, data_path, save_path, tickers, date_array, horizon, num_worker=args.worker)