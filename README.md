This repository contains the pre-release code for the (F)TGCN-based quantile and mean models as presented in our paper, "Big Portfolio Selection by Graph-Based Conditional Moments Method." You can access the paper on [Arxiv](https://arxiv.org/abs/2301.11697).

## Environment

- **Main Settings:** Python 3.9 & Pytorch 1.11.0 & CUDA 10.2
- **Minor Settings:** To be completed.

## Data

- **Stock Data:** The price and volume data for each stock, sector-industry relation data, and wiki relation data can be downloaded from the official repository of Feng (2019); see the [stock data repository](https://github.com/hennande/Temporal_Relational_Stock_Ranking/tree/master/data).
- **Factor Data:** Daily Fama-French five factors can be downloaded from the homepage of Kenneth R. French; see the [factor data download link](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Data Pipeline

| Script                     | Description                                                                 |
| :------------------------: | :--------------------------------------------------------------------------: |
| `compute_factor_loading.py` | Calculates factor loadings from raw End-of-Day data and factor data.         |
| `construct_feature.py`      | Generates the network input (including lagged values) for each trading day.  |
| `construct_label.py`        | Generates the label for each trading day.                                    |

## Network

| Script                     | Description                                                                 |
| :------------------------: | :--------------------------------------------------------------------------: |
| `model.py`                 | Specifies the model architecture of the network.                             |
| `my_dataset.py`            | Defines the dataset structure based on PyTorch.                              |
| `load_data.py`             | Loads the relation data.                                                     |
| `(F)TGCN.py`               | Implements the agent used for training the (F)TGCN.                          |
| `train_(F)TGCN.py`         | Trains the (F)TGCN-based quantile (mean) model.                              |
| `hypothesis_test.py`       | Performs the Kupiec and Christofer tests.                                    |
| `QCM.py`                   | Implements QCM learning from conditional quantiles.                          |
| `inference_(F)TGCN.py`     | Obtains four moments from the trained models.  

## Reproduce the Results for NASDAQ-Wikidata with FTGCN Models

```bash
# Ensure you have updated the data path and log directory in each file.

# Step 1: Construct features and labels
python compute_factor_loading.py
python construct_feature.py
python construct_label.py

# Step 2: Train models
# Mean model 
python train_FTGCN.py --tau 0.0 --mse-loss --lam 0.1 
# Quantile models
python train_FTGCN.py --tau 0.005 --lam 0.1 
python train_FTGCN.py --tau 0.01 --lam 0.1 
...
python train_FTGCN.py --tau 0.99 --lam 0.1 
python train_FTGCN.py --tau 0.995 --lam 0.1

# Step 3: Inference and QCM learning
python inference_FTGCN.py
```

## Cite
If you find this code helpful, please consider citing our paper:
```
@article{zhu2023big,
  title={Big portfolio selection by graph-based conditional moments method},
  author={Zhu, Zhoufan and Zhang, Ningning and Zhu, Ke},
  journal={arXiv preprint arXiv:2301.11697},
  year={2023}
}
```
## Contact
Please feel free to raise an issue in this GitHub repository or email me if you have any questions or encounter any issues.