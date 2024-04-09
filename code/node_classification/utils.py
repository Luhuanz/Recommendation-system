import networkx as nx
from torch_scatter import scatter
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops, k_hop_subgraph, degree, to_networkx
#to_scipy_sparse_matrix 将PyTorch Geometric的图数据转换为SciPy的稀疏矩阵格式。
#k_hop_subgraph提取与给定节点集合在k跳内相连的子图。这可以用于分析节点的局部邻域或者在图中执行局部操作。
#to_networkx: 将PyTorch Geometric的图数据转换为NetworkX图。这使得可以利用NetworkX提供的丰富图算法和可视化工具。
from scipy.sparse import csr_matrix
import random
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
import os


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

#旨在计算图中边的Jaccard相似度，并将计算结果保存为一个文件。Jaccard相似度是一种度量集合相似度的方法
def cal_jc(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        jacard_simi = (nei_nei_cap / (nei_nei_cup - nei_nei_cap)).mean(dim=1)

        edge_weight[i, neighbors] = jacard_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_jc.pt')

    return edge_weight


#旨在计算图中每个节点及其邻居之间的结构相似度，并将这种相似度作为边权重。
# 这个过程涉及到创建一个邻接矩阵、计算每个节点的邻居之间的结构相似度，然后基于这些相似度调整边权重。 "structural similarity"，即“结构相似度
def cal_sc(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0]) #初始化一个和邻接矩阵相同大小的边权重矩阵edge_weight，用于存储计算出的结构相似度作为边权重。
    adj_matrix[edge_index[0], edge_index[1]] = 1  #使用edge_index填充邻接矩阵，如果节点i到节点j有边，则在相应位置[i, j]放置1
    edge_weight = torch.zeros((x.shape[0], x.shape[0]))#torch.Size([2708, 2708])
    #计算结构相似度

    for i in range(x.shape[0]):
        # print( adj_matrix[i, :].nonzero()) #获取对应不为0位置索引
        # print(adj_matrix[i, :].nonzero().size())
        # exit()
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)  #[0,633..]# 去掉维度
        neighbors_adj = adj_matrix[neighbors] #获取指定节点i的所有邻居节点的邻接矩阵，#torch.Size([4, 2708])
        # 目的为了后续计算节点i的邻居之间的结构相似度做准备。
        # 具体来说，通过分析neighbors_adj，我们可以了解节点i的邻居之间是否也相互连接，以及它们与图中其他节点的连接情况。
        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t()) #反映了两个邻居之间共同邻居的数量
        nei_nei_cup = neighbors_adj.sum(dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1) #计算了每个邻居节点的度数（即每个节点连接的边数），然后将每个节点的度数与自己的度数相加

        sc_simi = (nei_nei_cap / ((neighbors_adj.sum(dim=1) *neighbors_adj.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1) #tensor([0.7302, 0.5493, 0.6972, 0.6677])
        edge_weight[i, neighbors] = sc_simi #torch.Size([2708, 2708]) 把第i个节点求得的sc_simi算出来的放到neighbors（索引）位置
        # print(edge_weight[edge_index[1], edge_index[0]].nonzero().squeeze(-1)) #tensor([ 2569,  7565, 10306, 10556])
    # print(edge_index[1].shape)#torch.Size([13264])
    edge_weight = edge_weight[edge_index[1], edge_index[0]] #   不是     无向图
    torch.save(edge_weight, os.getcwd() + '/data/' +     #边的权重写入pt中
               args.dataset + '/cir_sc.pt')
    # print(edge_weight.shape) #torch.Size([13264])
    # exit()
    return edge_weight

#计算图中每对节点之间的Leicht-Holme-Newman (LHN) 相似度
def cal_lhn(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        lhn_simi = (nei_nei_cap / ((neighbors_adj.sum(dim=1) *
                                    neighbors_adj.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[i, neighbors] = lhn_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_lhn.pt')

    return edge_weight

#计算图中节点间的共现相似度
def cal_co(edge_index, x, args):
    adj_matrix = torch.zeros(x.shape[0], x.shape[0])
    adj_matrix[edge_index[0], edge_index[1]] = 1

    edge_weight = torch.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        neighbors = adj_matrix[i, :].nonzero().squeeze(-1)

        neighbors_adj = adj_matrix[neighbors]

        nei_nei_cap = torch.matmul(neighbors_adj, neighbors_adj.t())
        nei_nei_cup = neighbors_adj.sum(
            dim=1) + neighbors_adj.sum(dim=1).unsqueeze(-1)

        co_simi = nei_nei_cap.mean(dim=1)

        edge_weight[i, neighbors] = co_simi

    edge_weight = edge_weight[edge_index[1], edge_index[0]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/cir_co.pt')

    return edge_weight
