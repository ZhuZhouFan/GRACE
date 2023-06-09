import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import statsmodels.api as sm
from tqdm import tqdm

data_path = '/data/GRACE_data/overseas'
FF5_df = pd.read_csv(f'{data_path}/Fama_French_daily.csv')
ymd = pd.to_datetime(FF5_df['date'], format='%Y%m%d')
FF5_df['date'] = ymd.apply(lambda x: f'{x.year}-{x.month:02d}-{x.day:02d}')
FF5_df.set_index('date', inplace = True)
FF5_df.sort_index(inplace = True)

market_name = 'NASDAQ'
tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
tickers = np.genfromtxt(os.path.join(data_path, tickers_fname), dtype=str, delimiter='\t', skip_header=False)
print('#tickers selected:', len(tickers))

def calculate_factor_loading(regression_df):
    ols_model = sm.OLS(regression_df['ret'], regression_df[['interception'] + ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    ols_result = ols_model.fit()
    return ols_result.params[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values

def sub_job(ticker, market_name, FF5_df, save_path):
    stock_price_df = pd.read_csv(f'{data_path}/google_finance/{market_name}_{ticker}_30Y.csv')
    stock_price_df['date'] = np.nan
    for j in range(stock_price_df.shape[0]):
        stock_price_df.loc[j, 'date'] = stock_price_df.at[j, 'Unnamed: 0'].split(' ')[0]
    stock_price_df.set_index('date', inplace = True)
    stock_price_df.sort_index(inplace = True)
    stock_price_df['ret'] = stock_price_df['Close']/stock_price_df['Close'].shift(1) - 1
    stock_price_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] = FF5_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    
    stock_price_df['interception'] = 1.0
    factor_loading_list = [f'{x}_loading' for x in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    stock_price_df[[factor_loading_list]] = np.nan
    for i, date in enumerate(stock_price_df.index.values):
        try:
            stock_price_df.loc[date, factor_loading_list] = calculate_factor_loading(stock_price_df.iloc[i - 120:i, :])
        except Exception as e:
            continue
    stock_price_df.to_csv(f'{save_path}/{market_name}/{ticker}.csv', columns=['ret'] + factor_loading_list)
    
result = Parallel(n_jobs=10)(delayed(sub_job)
                                 (ticker, market_name, FF5_df, '/data/GRACE_data/kline_day')
                                 for ticker in tqdm(tickers))