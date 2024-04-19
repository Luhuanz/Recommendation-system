import networkx as nx
from networkx.algorithms import shortest_paths
from torch_scatter import scatter
import torch
import os
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops, k_hop_subgraph, degree, to_networkx
from scipy.sparse import csr_matrix
from networkx.algorithms.shortest_paths.unweighted import  single_source_shortest_path_length
import random
#to_scipy_sparse_matrix 将PyTorch Geometric的图数据转换为SciPy的稀疏矩阵格式。
#k_hop_subgraph提取与给定节点集合在k跳内相连的子图。这可以用于分析节点的局部邻域或者在图中执行局部操作。
#to_networkx: 将PyTorch Geometric的图数据转换为NetworkX图。这使得可以利用NetworkX提供的丰富图算法和可视化工具。

# def propagate(x, edge_index, edge_weight=None):
#     """ feature propagation procedure: sparsematrix
#     """
#     edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

#     # calculate the degree normalize term
#     row, col = edge_index
#     deg = degree(col, x.size(0), dtype=x.dtype)
#     deg_inv_sqrt = deg.pow(-0.5)
#     # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
#     if(edge_weight == None):
#         edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#     # normalize the features on the starting point of the edge
#     out = edge_weight.view(-1, 1) * x[row]

#     return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate(x, edge_index, edge_weight):
    """ feature propagation procedure: sparsematrix
    """
    row, col = edge_index

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row] #边权重与特征相乘

    return scatter(out, col, dim=0, dim_size=x.size(0), reduce='add') #


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#将训练集、验证集和测试集的索引转换成掩码（mask）形式，用于在图神经网络等场景中方便地选择对应的节点子集进行操作。
def index_to_mask(train_index, val_index, test_index, size):
    train_mask = torch.zeros(size, dtype=torch.bool)
    val_mask = torch.zeros(size, dtype=torch.bool)
    test_mask = torch.zeros(size, dtype=torch.bool)

    train_mask[train_index] = 1
    val_mask[val_index] = 1
    test_mask[test_index] = 1

    return train_mask, val_mask, test_mask

#从原始图数据中提取一个子图，这个子图是根据指定数量的训练节点及其k跳邻居构建的
def subgraph_extract(data, train_num, k_hop):
    # pyg.DATA中data节点特征（data.x）、边索引（data.edge_index）、节点标签（data.y）
    # 以及训练集、验证集、测试集的掩码（data.train_mask、data.val_mask、data.test_mask）
    train_node_idx = torch.tensor(np.random.choice(torch.where(
        data.train_mask == True)[0].numpy(), train_num, replace=False)) #这是所有训练节点（没有被mask的）的索引

    subnodes, sub_edge_index, node_mapping, edge_mapping = k_hop_subgraph(
        train_node_idx, k_hop, data.edge_index, relabel_nodes=True)

    sub_x = data.x[subnodes]
    sub_train_mask, sub_val_mask, sub_test_mask = data.train_mask[
        subnodes], data.val_mask[subnodes], data.test_mask[subnodes]

    sub_y = data.y[subnodes]

    return sub_x, sub_y, sub_edge_index, sub_train_mask, sub_val_mask, sub_test_mask, subnodes

#计算一个图中指定子节点集合到图中所有其他节点的最短路径长度， line/deepwalk
def shortest_path(data, subnodes):
    G = to_networkx(data)
    p_l = torch.zeros((len(subnodes), data.x.size(0)))

    for i in range(len(subnodes)):
        #从单一源节点到图中所有其他节点的最短路径长度
        #从哪个节点开始计算到其他所有节点的最短路径长度
        lengths = single_source_shortest_path_length(
            G, source=subnodes[i].item())
        for key in lengths:
            if(lengths[key] != 0):
                p_l[i, key] = 1 / lengths[key]
            else:
                p_l[i, key] = 1

    p_l = p_l.t()

    return p_l
