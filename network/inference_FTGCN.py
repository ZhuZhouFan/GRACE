import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from model import Graph_Network
import argparse
from QCM import compute_QCM_table
from load_data import load_FF5_augmented_relation_data


def isnumber(x):
    """Identify whether an object is a number

    Args:
        x (_type_): object

    Returns:
        bool: True of False
    """    
    try:
        float(x)
        return True
    except:
        return False
    
    
def load_model_weights(tau:float,
                       N:int,
                       P:int, 
                       rel_encoding:np.array,
                       rel_mask:np.array,
                       hidden_dim:int,
                       quantile_model_path:str,
                       device:str='cuda'):
    """Load the trained FTGCN model weights

    Args:
        N (int): number of nodes in the graph
        P (int): dimensionality of input
        rel_encoding (np.array): np.array used for represent the graph
        rel_mask (np.array): mask the self-connected edges
        hidden_dim (int): the dimensionality of hidden state vectors of GNN
        quantile_model_path (str): the path of trained quantile models
        device (str, optional): use cpu or gpu (cuda) to run inference. Defaults to 'cuda'.
        
    Returns:
        nn.module: network model
    """    
    network = Graph_Network(batch_size=N,
                            rel_encoding=rel_encoding,
                            rel_mask=rel_mask,
                            feature_dim=P,
                            units=hidden_dim
                            ).to(device)
    network.graph_layer.all_one = network.graph_layer.all_one.to(device)
    network.graph_layer.relation = network.graph_layer.relation.to(device)
    network.graph_layer.rel_mask = network.graph_layer.rel_mask.to(device)
    
    network.load_state_dict(torch.load(f'{quantile_model_path}/{tau}/network_best.pth', map_location=device))
    network.eval()
    
    return network


def inference(N:int,
              P:int,
              hidden_dim:int,
              rel_encoding:np.array,
              rel_mask:np.array,
              tau:float,
              date_array:np.array,
              stock_list:list,
              quantile_model_path:str,
              tensor_path:str,
              device:str='cuda'):
    T = date_array.shape[0]
    stock_num = N - 5 # 5 is the number of factor nodes
    
    network = load_model_weights(tau, N, P, rel_encoding, 
                                 rel_mask, hidden_dim,
                                 quantile_model_path, device)
    
    # initialize a dictionary to save the results of inference
    result_dict = dict.fromkeys(date_array, 0)
    for date in date_array:
        result_dict[date] = pd.DataFrame(columns=['date', 'c_code', tau, 'ground_truth'])
        result_dict[date]['c_code'] = stock_list
        result_dict[date]['date'] = date

    with torch.no_grad():
        for i in range(T):
            date = date_array[i]
            feature_tensor = np.load(f'{tensor_path}/{date}/feature.npy')
            X = torch.Tensor(feature_tensor).to(device)
            network_output = network.forward(X)
            # remove the output of factor nodes
            network_output = network_output[:stock_num, :]

            result_dict[date][tau] = network_output.cpu().numpy()
            try:
                label_tensor = np.load(f'{tensor_path}/{date}/label.npy')
                result_dict[date]['ground_truth'] = label_tensor[:stock_num, 0]
            except FileNotFoundError:
                result_dict[date]['ground_truth'] = np.nan

    result_df = pd.concat(result_dict.values(), axis=0)
    return result_df


def obtain_inference_df(P:int,
              hidden_dim:int,
              rel_encoding:np.array,
              rel_mask:np.array,
              date_array:np.array,
              tau_list:list,
              stock_list:list,
              quantile_model_path:str,
              tensor_path:str,
              device:str='cuda'):
    
    inference_dict = dict.fromkeys(tau_list, 0)
    for tau in tqdm(tau_list, desc='inference'):
        result_df = inference(N=len(stock_list) + 5,
                              P=P,
                              hidden_dim=hidden_dim,
                              rel_encoding=rel_encoding,
                              rel_mask=rel_mask,
                              tau=tau,
                              date_array=date_array,
                              stock_list=stock_list,
                              quantile_model_path=quantile_model_path,
                              tensor_path=tensor_path,
                              device=device)
        if tau != tau_list[-1]:
            inference_dict[tau] = result_df.set_index(['date', 'c_code'])[[tau]]
        else:
            inference_dict[tau] = result_df.set_index(['date', 'c_code'])[[tau, 'ground_truth']]
        
    inference_df = pd.concat(inference_dict.values(), axis = 1)
    inference_df.reset_index(inplace = True)
    return inference_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Id of device to use')
    parser.add_argument('--market', type=str, default='NASDAQ',
                        help='Which market dataset is used for training')
    parser.add_argument('--rel', type=str, default='wikidata',
                        help='Which relation data is used for training')
    parser.add_argument('--train-start-time', type=str, default='2013-01-02',
                        help='Training dataset start time')
    parser.add_argument('--train-end-time', type=str, default='2016-12-30',
                        help='Training dataset end time')
    parser.add_argument('--infer_end_time', type=str, default='2017-12-08',
                        help='Inference end time')
    parser.add_argument('--size', type=float, default=0.01,
                        help='Significance level')
    args = parser.parse_args()

    hidden_dim = 64
    lag_order = 16
    P = 10
    seed = 42
    torch.cuda.set_device(args.cuda)
    train_start_time = args.train_start_time
    train_end_time = args.train_end_time
    infer_end_time = args.infer_end_time

    data_path = 'Specify/Your/Data/Path/Here'
    model_path = f'{data_path}/FTGCN_model/{args.market}_{args.rel}/{train_start_time}_{train_end_time}/hidden_{hidden_dim}_lag_{lag_order}_horizon_1_seed_42'
    tensor_path = f'{data_path}/tensors/{args.market}'
    moment_path = f'{data_path}/FTGCN_moment/{args.market}_{args.rel}/{train_start_time}_{train_end_time}_{infer_end_time}/hidden_{hidden_dim}_lag_{lag_order}_horizon_1_seed_42'

    stock_list = np.genfromtxt(
        f'{data_path}/{args.market}_tickers_qualify_dr-0.98_min-5_smooth.csv', dtype=str, delimiter='\t', skip_header=False)

    if args.rel == 'wikidata':
        rel_encoding = load_FF5_augmented_relation_data(
            f'{data_path}/relation/wikidata/{args.market}_wiki_relation.npy')
    elif args.rel == 'sector_industry':
        rel_encoding = load_FF5_augmented_relation_data(
            f'{data_path}/relation/sector_industry/{args.market}_industry_relation.npy')

    rel_shape = [rel_encoding.shape[0], rel_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int), np.sum(rel_encoding, axis=2))
    rel_mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))

    if not os.path.exists(moment_path):
        os.makedirs(moment_path)

    tau_list = os.listdir(model_path)
    tau_list = [float(x) for x in tau_list if isnumber(x)]
    tau_list.sort()

    date_list = os.listdir(tensor_path)
    date_list.sort()
    date_array = np.array(date_list)
    date_array = date_array[(date_array >= train_start_time) & (date_array <= infer_end_time)]

    inference_df = obtain_inference_df(P, hidden_dim, rel_encoding, rel_mask,
                                       date_array, tau_list, stock_list,
                                       model_path, tensor_path)

    # valid_tau_path = f'{moment_path}/valid_tau'
    # if not os.path.exists(valid_tau_path):
    #     os.makedirs(valid_tau_path)

    for stock_name in tqdm(stock_list, desc='QCM regression'):
        
        try:
            moments_df, selected_tau_list = compute_QCM_table(stock_name,
                                                      inference_df,
                                                      tau_list,
                                                      tolerance=20,
                                                      size=args.size,
                                                      start_time=train_start_time,
                                                      valid_time=train_end_time)
            if (moments_df is None) or (selected_tau_list is None):
                continue
            else:
                moments_df.to_csv(f'{moment_path}/{stock_name}.csv')
                # np.save(f'{valid_tau_path}/{stock_name}.npy', selected_tau_list)
        except Exception as e:
            print(f'{stock_name} {e}')