import torch
import os



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
