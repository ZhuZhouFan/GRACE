import torch
import pandas as pd
import numpy as np
import argparse
from load_data import load_relation_data
from TGCN import TGCN_Agent

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    
    ## arguments related to loss function ##
    parser.add_argument('--mse-loss', action='store_true', default=False,
                        help='Use the MSE as the loss')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Quantile level')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='Tuning parameter for rank loss')
    
    ## arguments related to weight and bias initialisation ##
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    ## arguments related to changing the model ##
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units in encoder')
    
    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int, default=0,
                        help='Number of device training on')
    parser.add_argument('--market', type=str, default='NASDAQ',
                        help='Which dataset is used for training')
    parser.add_argument('--rel', type=str, default='wikidata',
                        help='Which relation data is used for training')
    parser.add_argument('--start-time', type=str, default='2013-01-02',
                        help='Training dataset start time')
    parser.add_argument('--valid-time', type=str, default='2015-12-31',
                        help='Training dataset end time (i.e., Validation dataset start time)')
    parser.add_argument('--end-time', type=str, default='2016-12-30',
                        help='Validation dataset end time')
    
    args = parser.parse_args()
    
    torch.cuda.set_device(args.cuda)
    start_time = args.start_time
    valid_time = args.valid_time
    end_time = args.end_time
    tau = args.tau
    num_workers = args.workers
    market_name = args.market
    relation_name = args.rel
    
    root_dir = 'Specify/Your/Data/Path/Here'
    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    tickers = np.genfromtxt(f'{root_dir}/{tickers_fname}', dtype=str, delimiter='\t', skip_header=False)
    
    lag_order = 16
    P = 10
    N = len(tickers)
    data_path = f'{root_dir}/tensors/{market_name}'
    rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
    
    rel_encoding = load_relation_data(f'{root_dir}/relation/{args.rel}/{market_name}{rname_tail[relation_name]}')
    
    log_dir_ = f'{root_dir}/TGCN_model/{market_name}_{relation_name}/{args.start_time}_{args.end_time}/hidden_{args.hidden}_lag_{lag_order}_horizon_1_seed_{args.seed}/{tau}'
    
    agent = TGCN_Agent(individual_num=N,
                       feature_dim=P,
                       hidden_dim=args.hidden,
                       rel_encoding = rel_encoding, 
                       log_dir=log_dir_,
                       learning_rate=args.lr,
                       seed=args.seed, 
                       patience=args.patience
                       )
    
    agent.load_data(data_path, start_time, valid_time, end_time, num_workers)
    agent.train(tau=tau, 
                epoch=args.epochs,
                lambda_=args.lam,
                mse_loss=args.mse_loss)