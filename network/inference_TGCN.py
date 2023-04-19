import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from model import Graph_Network
from QCM import compute_QCM_table
from load_data import load_FF5_augmented_relation_data


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


def inference(N, P, hidden_dim,
              rel_encoding, rel_mask,
              tau, date_array, stock_list,
              model_path, tensor_path,
              device='cuda'):
    T = date_array.shape[0]
    network = Graph_Network(batch_size=N,
                            rel_encoding=rel_encoding,
                            rel_mask=rel_mask, 
                            feature_dim=P,
                            units=hidden_dim
                            ).to(device)
    network.graph_layer.all_one = network.graph_layer.all_one.to(device)
    network.graph_layer.relation = network.graph_layer.relation.to(device)
    network.graph_layer.rel_mask = network.graph_layer.rel_mask.to(device)
    
    network.load_state_dict(torch.load(
        f'{model_path}/{tau}/network_best.pth', map_location=device))
    network.eval()

    result_dict = dict.fromkeys(date_array, 0)
    mat_dict = dict.fromkeys(date_array, 0)

    for date in date_array:
        result_dict[date] = pd.DataFrame(
            columns=['date', 'c_code', tau, 'ground_truth'])
        result_dict[date]['c_code'] = stock_list
        result_dict[date]['date'] = date

    with torch.no_grad():
        for i in range(T):
            date = date_array[i]
            feature_tensor = np.load(f'{tensor_path}/{date}/feature.npy')
            X = torch.Tensor(feature_tensor[:N, :, :]).to(device)            
            network_output = network.forward(X)
            result_dict[date][tau] = network_output.cpu().numpy()
            try:
                label_tensor = np.load(f'{tensor_path}/{date}/label.npy')
                result_dict[date]['ground_truth'] = label_tensor[:, 0]
            except FileNotFoundError:
                result_dict[date]['ground_truth'] = np.nan

    result_df = pd.concat(result_dict.values(), axis=0)
    return result_df


hidden_dim = 64
P = 10
torch.cuda.set_device(0)
device = 'cuda'
data_path = '/data/GRACE_data/overseas'
model_path = f'{data_path}/FTGCN_models/hidden_128_lag_24_horizon_1'
tensor_path = f'{data_path}/tensors/NASDAQ/lag_24_horizon_1'
moment_path = f'{data_path}/moments/NASDAQ/wikidata/'
start_time = '2013-01-02'
valid_time = '2016-12-30'
end_time = '2017-12-08'
stock_list = np.genfromtxt('/data/GRACE_data/overseas/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv', dtype=str, delimiter='\t', skip_header=False)
rel_encoding = load_FF5_augmented_relation_data('/data/GRACE_data/overseas/relation/wikidata/NASDAQ_wiki_relation.npy')
rel_shape = [rel_encoding.shape[0], rel_encoding.shape[1]]
mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                  np.sum(rel_encoding, axis=2))
rel_mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))

if not os.path.exists(moment_path):
    os.makedirs(moment_path)

tau_list = os.listdir(model_path)
tau_list = [float(x) for x in tau_list if isnumber(x)]
tau_list.sort()

date_list = os.listdir(tensor_path)
date_list.sort()
date_array = np.array(date_list)
date_array = date_array[(date_array >= start_time) & (date_array <= end_time)]

inference_dict = dict.fromkeys(tau_list, 0)
for tau in tqdm(tau_list, desc='inference'):
    result_df, mat_dict, K = inference(N = len(stock_list) + 5,
                                       P=P,
                                       hidden_dim=hidden_dim,
                                       rel_encoding = rel_encoding,
                                       rel_mask = rel_mask, 
                                       tau=tau,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=model_path,
                                       tensor_path=tensor_path,
                                       device=device)
    inference_dict[tau] = result_df

valid_tau_path = f'{moment_path}/valid_tau'
if not os.path.exists(valid_tau_path):
    os.makedirs(valid_tau_path)

for stock_name in tqdm(stock_list, desc='QCM regression'):
    try:
        df, selected_tau_list = compute_QCM_table(stock_name,
                                               inference_dict,
                                               tau_list,
                                               tolerance=20,
                                               size=0.01,
                                               start_time=start_time,
                                               valid_time=valid_time)
        if (df is None) or (selected_tau_list is None):
            continue
        else:
            df.to_csv(f'{moment_path}/{stock_name}.csv')
            np.save(f'{valid_tau_path}/{stock_name}.npy', selected_tau_list)
    except Exception as e:
        print(f'{stock_name} {e}')