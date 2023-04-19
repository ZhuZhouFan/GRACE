Here is the pre-released code for the FTGCN-based quantile and mean models in our paper "Big portfolio selection by graph-based conditional moments method", [\[paper\]](https://arxiv.org/abs/2301.11697).

## Environment
Main settings: Python 3.9 & Pytorch 1.11.0

Minor settings: To complete.

## Data

The price and volume Data of each stock, sector-industry relation data, and wiki relation data, could be downloaded from the official repositiy of Feng (2019); see [\[stock data\]](https://github.com/hennande/Temporal_Relational_Stock_Ranking/tree/master/data).

In the meanwhile, the daily Fama French five factors could be downloaded from the homepage of Kenneth R. French; see [\[factor data\]](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## data_pipe
| Script | Usage |
| :-----------: | :-----------: |
| compute_factor_loading.py | To calculate factor loadings from raw End-of-day data and factor data|
| construct_feature.py | Generate the network input (including lagged values) for each day|
| construct_label.py | Generate the label for each day|

## network
| Script | Usage |
| :-----------: | :-----------: |
| model.py | The model specification of network |
| my_dataset.py | The dataset specification based on Pytorch |
| load_data.py | Load the relation data |
| (F)TGCN.py | The agent used for training (F)TGCN |
| train_(F)TGCN.py | Train a model of (F)TGCN-based quantile (mean) model |
| hypothesis_test.py | The Kupiec and Christofer tests |
| QCM.py | The QCM learning from conditional quantiles |
| inference_(F)TGCN.py | Obtain four moments from the trained models |


## Reproduce the results for NASDAQ-wikidata with FTGCN
```
# Please make sure you have changed the log directory in each file.

# Construct features and labels
python compute_factor_loading.py
python construct_feature.py
python construct_label.py

# Train models
# mean model 
python train_FTGCN.py --tau 0.0 --mse-loss --lam 0.1 --save_folder ... 
# quantile models
python train_FTGCN.py --tau 0.005 --lam 0.1 --save_folder ...
python train_FTGCN.py --tau 0.01 --lam 0.1 --save_folder ...  
...
python train_FTGCN.py --tau 0.99 --lam 0.1 --save_folder ... 
python train_FTGCN.py --tau 0.995 --lam 0.1 --save_folder ... 

# Inference and QCM learning
python inference_FTGCN.py
```

## Cite
If you feel this code helps, please kindly cite the following paper:
```
@article{zhu2023big,
  title={Big portfolio selection by graph-based conditional moments method},
  author={Zhu, Zhoufan and Zhang, Ningning and Zhu, Ke},
  journal={arXiv preprint arXiv:2301.11697},
  year={2023}
}
```
