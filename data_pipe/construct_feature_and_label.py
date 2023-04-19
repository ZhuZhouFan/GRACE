import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

def isnumber(x):
    try:
        float(x)
        return False
    except:
        return True

def extract_data(stock_name, 
                 date,
                 previous_date,
                 horizon,
                 data_path):
    
    stock_df = pd.read_csv(f'{data_path}/{stock_name}.csv', index_col='date')
    stock_df['label'] = stock_df['close'].shift(-1 * horizon)/stock_df['close'] - 1
    # stock_df['total_turnover'] = np.log(stock_df['total_turnover'] + 1e-6)
    # stock_df['volume'] = np.log(stock_df['volume'] + 1e-6)
    # stock_df['a_share_market_val_in_circulation'] = np.log(stock_df['a_share_market_val_in_circulation'] + 1e-6)

    # feature_list = ['open', 'close', 'high', 'low', 'num_trades', 'total_turnover', 'volume', 'log_ret',
    #                 'a_share_market_val_in_circulation', 'du_return_on_equity_ttm', 'inc_revenue_ttm',
    #                 'total_asset_turnover_ttm', 'debt_to_asset_ratio_ttm']
    feature_list = ['ret', 'ret_5', 'ret_10', 'ret_20', 'ret_30']
    factor_loading_list = ['Mkt-RF_loading', 'SMB_loading', 'HML_loading', 'RMW_loading', 'CMA_loading']
    
    feature = pd.DataFrame(columns=feature_list + factor_loading_list, index = stock_df.index)
    
    feature = stock_df.loc[:, feature_list + factor_loading_list]
    feature = (feature - feature.min())/(feature.max() - feature.min())
    
    feature.fillna(method = 'ffill', inplace=True)
    feature.fillna(0.0, inplace=True)
    feature = feature.loc[previous_date:date, :]
    
    try:
        label = stock_df.at[date, 'label']
        if (np.isnan(label)) or (np.isinf(label)):
            label = 0.0
    except KeyError as e:
        label = 0.0
    
    return feature, label

def one_day(date, data_path, save_path, horizon, lag_order, index_components_df, num_worker=20):
    previous_date = date_array[np.where(date_array == date)[0].item() - lag_order + 1]
    index_components = index_components_df.loc[date, :].values

    one_day_list = Parallel(n_jobs=num_worker)(delayed(extract_data)
                                               (stock_name, date, previous_date, horizon, data_path,)
                                               for stock_name in index_components)

    feature_tensor = np.zeros([index_components.shape[0], lag_order, P])
    label_tensor = np.zeros([index_components.shape[0], 1])

    for i in range(len(one_day_list)):
        if one_day_list[i][0].values.shape[0] < lag_order:
            try:
                tem = one_day_list[i][0].values.shape[0]
                feature_tensor[i, -tem:, :] = one_day_list[i][0].values
            except Exception as e:
                print(date, index_components[i])
        else:
            feature_tensor[i, :, :] = one_day_list[i][0].values
        label_tensor[i, 0] = one_day_list[i][1]

    np.save(f'{save_path}/{date}/feature.npy', feature_tensor)
    np.save(f'{save_path}/{date}/label.npy', label_tensor)

data_path = '/data/GRACE_data/A_share'
start_time = '2015-01-01'
end_time = '2023-03-01'
lag_order = 16
P = 10
horizon = 1
save_path = f'{data_path}/tensors/FF5/lag_{lag_order}_horizon_{horizon}'

index_components_df = pd.read_csv(f'{data_path}/stock_pool.csv', index_col='date')
index_components_df[['fac_1', 'fac_2', 'fac_3', 'fac_4', 'fac_5']] = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
index_data = pd.read_csv(f'{data_path}/index_kline_day/000001.XSHG.csv', index_col='date')
date_array = (index_data.loc[start_time: end_time, :].index.values)
stock_path = f'{data_path}/daily_feature'

for date in tqdm(date_array[lag_order:-1], desc='construct feature and label'):
    date_save_path = f'{save_path}/{date}'
    if (os.path.exists(f'{date_save_path}/feature.npy') & os.path.exists(f'{date_save_path}/label.npy')):
        continue
    elif not os.path.exists(date_save_path):
        os.makedirs(date_save_path)

    one_day(date, stock_path, save_path, horizon, lag_order, index_components_df, num_worker=30)