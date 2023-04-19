import torch
import pandas as pd
import numpy as np
import argparse

from FTGCN import FTGCN_Agent

parser = argparse.ArgumentParser()
## arguments related to training ##
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Initial learning rate.')
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
                    help='tuning parameter for rank loss.')

## arguments related to weight and bias initialisation ##
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

## arguments related to changing the model ##
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units in encoder.')

## Saving, loading etc. ##
parser.add_argument('--cuda', type=int, default=0,
                    help='Number of device training on.')
parser.add_argument('--save-folder', type=str, default='/data/GRACE_data/A_share/FTGCN_model',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--start-time', type=str, default='2015-01-01',
                    help='The beginning of training dataset')
parser.add_argument('--valid-time', type=str, default='2019-01-01',
                    help='The ending of training dataset')
parser.add_argument('--end-time', type=str, default='2021-01-01',
                    help='The ending of ending dataset')

args = parser.parse_args()

lag_order = 16
P = 10
N = 1682 + 5
root_dir = '/data/GRACE_data/A_share'
data_path = f'{root_dir}/tensors/FF5/lag_{lag_order}_horizon_1'

torch.cuda.set_device(args.cuda)
start_time = args.start_time
valid_time = args.valid_time
end_time = args.end_time
log_dir = args.save_folder
tau = args.tau
num_workers = args.workers

rel_encoding = np.load(f'{root_dir}/factor_augmented_sector_ind_graph.npy')

log_dir_ = f'{log_dir}/hidden_{args.hidden}_lag_{lag_order}_horizon_1_seed_{args.seed}/{tau}'

agent = FTGCN_Agent(individual_num=N,
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