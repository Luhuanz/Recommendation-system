import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn import global_mean_pool
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor,
                                    Size)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
def DAconv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "add" stands for sum aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)
    def message(self, x_j, norm):
        #  消息函数处理节点j的特征x_j
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 更新函数应用线性变换到聚合的输出上
        return self.lin(aggr_out)

    def propagate(self,edge_index,size,x,norm):
        return self.message(x_j=self.index_select(edge_index, 0), norm=norm)
