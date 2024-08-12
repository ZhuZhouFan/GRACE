import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphModule(nn.Module):
    def __init__(self,
                 batch_size:int,
                 fea_shape:int,
                 rel_encoding:np.array,
                 rel_mask:np.array,
                 inner_prod = False):
        """Graph convolution module

        Args:
            batch_size (int): batch_size
            fea_shape (int): the length of embedding computed by LSTM
            rel_encoding (np.array): the encoding of relation
            rel_mask (np.array): masked signal
            inner_prod (bool, optional): whether inner product the weights. Defaults to False.
        """        
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = fea_shape
        self.inner_prod = inner_prod
        self.relation = nn.Parameter(torch.Tensor(rel_encoding),
                                     requires_grad=False)
        self.rel_mask = nn.Parameter(torch.Tensor(rel_mask),
                                     requires_grad=False)
        self.all_one = nn.Parameter(torch.ones(self.batch_size, 1),
                                    requires_grad=False)
        self.rel_weight = nn.Linear(rel_encoding.shape[2], 1)
        if self.inner_prod is False:
            self.head_weight = nn.Linear(fea_shape, 1)
            self.tail_weight = nn.Linear(fea_shape, 1)

    def forward(self, inputs):
        rel_weight = self.rel_weight(self.relation)
        if self.inner_prod:
            inner_weight = inputs @ inputs.t().contiguous()
            weight = inner_weight @ rel_weight[:, :, -1]
        else:
            all_one = self.all_one
            head_weight = self.head_weight(inputs)
            tail_weight = self.tail_weight(inputs)
            weight = (head_weight @ all_one.t().contiguous() +
                      all_one @ tail_weight.t().contiguous()) + rel_weight[:, :, -1]
        weight_masked = F.softmax(self.rel_mask + weight, dim=0)
        outputs = weight_masked @ inputs
        return outputs


class Graph_Network(nn.Module):
    def __init__(self, 
                 feature_dim:int,
                 batch_size:int,
                 rel_encoding:np.array,
                 rel_mask:np.array, 
                 units:int = 64,
                 inner_prod = False):
        super().__init__()
        self.batch_size = batch_size
        # here the feature_dim denotes the dimension of each stock feature
        self.lstm = nn.LSTM(feature_dim, units, batch_first=True)
        self.graph_layer = GraphModule(batch_size = batch_size,
                                       fea_shape = units,
                                       rel_encoding = rel_encoding,
                                       rel_mask = rel_mask, 
                                       inner_prod = inner_prod)
        self.fc = nn.Linear(units * 2, 1)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        outputs_graph = self.graph_layer(x)
        outputs_cat = torch.cat([x, outputs_graph], dim=1)
        prediction = self.fc(outputs_cat)
        return prediction