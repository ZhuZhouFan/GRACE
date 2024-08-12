'''
To load the relation data provided by Feng (2019), we uses the function provided by its official github respository.

https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/blob/master/training/load_data.py
'''

import copy
import numpy as np
import os

def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    # rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
    #                       np.sum(relation_encoding, axis=2))
    # mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    # return relation_encoding, mask
    return relation_encoding

def load_FF5_augmented_relation_data(relation_file):
    stock_relation_encoding = np.load(relation_file)
    stock_rel_type = stock_relation_encoding.shape[-1]
    stock_num = stock_relation_encoding.shape[0]
    # augment the hypergraph with additional 5 nodes and 5 relations
    relation_encoding = np.zeros([stock_num + 5, stock_num + 5, stock_rel_type + 5])
    relation_encoding[:stock_num, :stock_num, :stock_rel_type] = stock_relation_encoding
    for k in range(5):
        relation_encoding[stock_num + k, :stock_num, (stock_rel_type + k)] = 1
        relation_encoding[:stock_num, stock_num + k, (stock_rel_type + k)] = 1
        relation_encoding[stock_num + k, stock_num + k, (stock_rel_type + k)] = 0
    # rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
    #                       np.sum(relation_encoding, axis=2))
    # mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    # return relation_encoding, mask
    return relation_encoding