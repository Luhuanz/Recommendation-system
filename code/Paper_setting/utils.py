import numpy as np
import random
import torch
import scipy.sparse as sp
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter_max, scatter_add
import seaborn as sns
import matplotlib.pyplot as plt
import os


def neg_sample_before_epoch(train_cf, clicked_set, args):
    #生成一个随机整数矩阵 neg_cf，其形状为训练数据集 train_cf 的行数乘以 args.K
    neg_cf = np.random.randint(
        args.n_users, args.n_users + args.n_items, (train_cf.shape[0], args.K))
    #遍历 train_cf 的每一行，即每一个用户
    for i in range(train_cf.shape[0]):
        # 从 clicked_set 字典中获取第 i 个用户已经点击过的物品集合
        user_clicked_set = clicked_set[train_cf[i, 0]]
        #对于每个用户，遍历 K 个负样本
        for j in range(args.K):
            #检查如果负样本 neg_cf[i, j] 已经被该用户点击过，则继续寻找新的负样本。
            while(neg_cf[i, j] in user_clicked_set):
                #如果当前负样本已经被点击过，则重新随机生成一个新的负样本，直到这个负样本没有被用户点击过。
                neg_cf[i, j] = np.random.randint(
                    args.n_users, args.n_users + args.n_items)
#返回生成的负样本矩阵
    return neg_cf


def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

#根据输入的邻接矩阵构建一个稀疏的KNN邻接矩阵。
def knn_adj(adj_sp_norm, args):
    #将输入的稀疏邻接矩阵 adj_sp_norm 转换为稠密矩阵
    adj_sp_norm = adj_sp_norm.to_dense()
    #使用 torch.topk 函数获取每行中最大的 args.knn 个元素，这些元素代表与每个节点最近的 k 个邻居。
    top_adj_sp_norm, _ = torch.topk(adj_sp_norm, args.knn)
    # 从 top_adj_sp_norm 中获取每行的最小值（即第 knn 个最大值）和最大值（即第一个最大值），分别存储在 low 和 high 中。
    low, high = top_adj_sp_norm[:, -1], top_adj_sp_norm[:, 0]
    #创建一个掩码矩阵，其中的元素根据 adj_sp_norm 是否在 low 和 high 的范围内来决定，以此来保留每个节点的 k 个最近邻居。
    mask = ((adj_sp_norm >= low.unsqueeze(1)) *
            (adj_sp_norm <= high.unsqueeze(1))).float()
    #应用掩码矩阵，将不是 k 个最近邻居的元素置为 0
    adj_sp_norm = adj_sp_norm * mask
    #取 adj_sp_norm 的上三角部分（不包括对角线），这样做是为了保证邻接矩阵的对称性，并且避免自环
    adj_sp_norm = torch.triu(adj_sp_norm, diagonal=1)
#找出 adj_sp_norm 中非零元素的索引，这些索引代表边的两个端点。
    edge_index = adj_sp_norm.nonzero()
#创建一个新的稀疏张量，表示 KNN 邻接矩阵。这里使用非零元素的位置和值来构造这个稀疏张量，并且设定其形状为 (args.n_users + args.n_items, args.n_users + args.n_items)
    adj_sp_norm = torch.sparse.FloatTensor(
        edge_index.t(), adj_sp_norm[edge_index[:, 0], edge_index[:, 1]], (args.n_users + args.n_items, args.n_users + args.n_items))

    return adj_sp_norm

#根据用户与物品的交互数据（train_cf），得到每个用户在总交互中所占的比例。
def ratio(train_cf, n_users):
    #初始化一个张量 user_link_num，其大小为 n_users（用户的数量）。这段代码通过遍历所有用户来计算每个用户的交互数。
    user_link_num = torch.tensor(
        [(train_cf[:, 0] == i).sum() for i in range(n_users)])

    link_ratio = n_users / user_link_num
    link_ratio = link_ratio / link_ratio.sum()
#其中包含与 train_cf 中每行对应用户 ID 的 link_ratio 值
    return link_ratio[train_cf[:, 0]]

#计算贝叶斯个性化排序（BPR）损失，通常用于推荐系统中优化个性化排序任务。
def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    pos_scores = torch.sum(
        torch.mul(user_embs, pos_item_embs), axis=1)

    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(
        dim=1), neg_item_embs), axis=-1)

    bpr_loss = torch.mean(torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1)))

    # bpr_loss = (link_ratios*torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1))).mean()

    return bpr_loss


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size


def softmax(src, index, num_nodes):
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()

    # out = src #method 2

    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out

#计算了基于用户和物品之间的交互的 Jaccard 相似度，并将其用作图中边的权重。
def co_ratio_deg_user_jacard(adj_sp_norm, edge_index, degree, args):
    #将邻接矩阵转换为稠密格式，并只选取用户和物品之间的关系部分，然后将其移动到 CPU 内存。
    user_item_graph = adj_sp_norm.to_dense()[:args.n_users, args.n_users:].cpu()
    #二值化处理，将用户 - 物品交互图中所有非零元素设置为 1，表示存在交互。
    user_item_graph[user_item_graph > 0] = 1
    #获得物品 - 用户图，即将用户 - 物品图进行转置
    item_user_graph = user_item_graph.t()
    #初始化边权重矩阵，尺寸为总用户数加总物品数的正方形矩阵
    edge_weight = torch.zeros((args.n_users + args.n_items, args.n_users + args.n_items))

    # 对于每个物品，找出与之交互的所有用户
    # 计算这些用户之间的交互物品集合的交集和并集，以计算Jaccard相似度。
    # 对于每个用户，找出与其交互的所有物品，并重复上述计算步骤
    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)
        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        jacard_simi = (user_user_cap / (user_user_cup -
                                        user_user_cap)).mean(dim=1)
        #更新边权重矩阵，将计算得到的 Jaccard 相似度作为相应边的权重
        edge_weight[users, i + args.n_users] = jacard_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        jacard_simi = (item_item_cap / (item_item_cup -
                                        item_item_cap)).mean(dim=1)

        edge_weight[items + args.n_users, i] = jacard_simi

    # print(edge_weight.nonzero().shape) 根据 edge_index 筛选最终的边权重，只保留原图中存在的边的权重。
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_jc.pt')
    #返回最终的边权重
    return edge_weight


def co_ratio_deg_user_jacard_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        jacard_simi = (user_user_cap / (user_user_cup -
                                        user_user_cap)).mean(dim=1)

        edge_weight[users, i + args.n_users] = jacard_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        jacard_simi = (item_item_cap / (item_item_cup -
                                        item_item_cap)).mean(dim=1)

        edge_weight[items, i] = jacard_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_jc.pt')

    return edge_weight

#计算了基于 Jaccard 相似度的用户与物品间边的权重，但与前一个函数不同的是，它处理稀疏数据结构
def co_ratio_deg_user_common(adj_sp_norm, edge_index, degree, args):
    user_item_graph = adj_sp_norm.to_dense()[:args.n_users, args.n_users:].cpu()
    user_item_graph[user_item_graph > 0] = 1
    item_user_graph = user_item_graph.t()
    #初始化边权重矩阵，大小为用户数加物品数的正方形矩阵，用于存储计算出的权重。
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))
    #对于每个物品 i ，找出与之相连的所有用户 users
    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        common_simi = user_user_cap.mean(dim=1)

        edge_weight[users, i + args.n_users] = common_simi
    #对于每个用户 i ，找出与其相连的所有物品 items
    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        common_simi = item_item_cap.mean(dim=1)

        edge_weight[items + args.n_users, i] = common_simi

    # print(edge_weight.nonzero().shape) 使用 edge_index 过滤出原始图中存在的边的权重，移除不需要的权重值
    edge_weight = edge_weight[edge_index[0], edge_index[1]]
    #将计算的边权重保存到文件中
    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_co.pt')
    #返回边权重矩阵。
    return edge_weight


def co_ratio_deg_user_common_sp(adj_sp_norm, edge_index, degree, args):
    #初始化一个边权重矩阵 edge_weight，其大小足以包含用户和物品之间的所有潜在连接。
    edge_weight = torch.zeros((args.n_users + args.n_items, args.n_users + args.n_items)) #
    #开始遍历物品，对于每个物品 i，计算它与用户的相似度
    for i in range(args.n_items):
        #通过 edge_index 找出与物品 i 相连的所有用户。这里，edge_index[1] == i + args.n_users 确定了物品 i 的索引，然后通过 edge_index[0] 获取相应的用户索引。
        users = edge_index[0, edge_index[1] == i + args.n_users]
        #初始化一个矩阵 items，用于存储与这些用户相连接的物品。矩阵的每一行对应一个用户，每一列对应一个物品。
        items = torch.zeros([users.shape[0], args.n_items])
        #遍历与当前物品相连的所有用户
        for j, user in enumerate(users):
    #填充 items 矩阵，对于每个用户，标记他们交互过的物品。这通过从用户数据集中获取每个用户的交互物品列表并在 items 矩阵中将对应位置设为 1 来实现
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1
    #计算用户之间的共同物品数量，通过矩阵乘法 items 和其转置 items.t() 实现
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)
    #计算每个用户对于当前物品的平均共有特征（或共同物品数量）作为相似度
        common_simi = user_user_cap.mean(dim=1)
    #更新 edge_weight 矩阵，为当前物品与其用户之间的连接设置权重。
        edge_weight[users, i + args.n_users] = common_simi
    #遍历用户，对于每个用户 i，计算它与物品的相似度。
    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        common_simi = item_item_cap.mean(dim=1)

        edge_weight[items, i] = common_simi

    # print(edge_weight.nonzero().shape) 使用 edge_index 筛选出原始图中存在的边，保留这些边的权重。
    edge_weight = edge_weight[edge_index[0], edge_index[1]]
    #将计算得到的边权重矩阵保存到文件。
    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_co.pt')

    return edge_weight

#计算了基于余弦相似度的边权重，用于表示用户和物品之间的关系强度。
def co_ratio_deg_user_sc(adj_sp_norm, edge_index, degree, args):
    #将稀疏邻接矩阵 adj_sp_norm 转换为稠密格式，并只取用户到物品的关系部分，然后将其移动到 CPU 内存。
    user_item_graph = adj_sp_norm.to_dense()[:args.n_users, args.n_users:].cpu()
    #二值化处理，将用户 - 物品关系图中所有非零元素设置为 1，表示存在交互。
    user_item_graph[user_item_graph > 0] = 1
    #获得物品 - 用户图，即将用户 - 物品图进行转置。
    item_user_graph = user_item_graph.t()
    #初始化边权重矩阵，其大小为用户和物品总数的平方
    edge_weight = torch.zeros((args.n_users + args.n_items, args.n_users + args.n_items))
    # 对于每个物品i，找出与之交互的所有用户users。构建一个表示这些用户交互的物品集的矩阵items。
    for i in range(args.n_items):
        users = user_item_graph[:, i].nonzero().squeeze(-1)

        items = user_item_graph[users]
        #计算用户间共享物品的数量（user_user_cap）和计算余弦相似度所需的分母部分（user_user_cup）。
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)
#  使用共享物品数量和分母部分来计算余弦相似度(sc_simi)。

        sc_simi = (user_user_cap / ((items.sum(dim=1) *
                                     items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)
#将计算得到的相似度作为边权重。
        edge_weight[users, i + args.n_users] = sc_simi


    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        sc_simi = (item_item_cap / ((users.sum(dim=1) *
                                     users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[items + args.n_users, i] = sc_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_sc.pt')

    return edge_weight

#计算基于余弦相似度的边权重，主要用于处理稀疏矩阵数据，适用于大规模用户-物品交互数据co_ratio 连接（degree）或共有（common）的比率。 sc：代表使用余弦相似度（Sine Cosine Similarity）计算。
def co_ratio_deg_user_sc_sp(adj_sp_norm, edge_index, degree, args):
    #初始化边权重矩阵，尺寸为用户数加物品数
    edge_weight = torch.zeros((args.n_users + args.n_items, args.n_users + args.n_items))
    #遍历每个物品以计算与用户之间的相似度
    for i in range(args.n_items):
        #从 edge_index 中找出与物品 i 相连的所有用户索引。
        users = edge_index[0, edge_index[1] == i + args.n_users]
        #初始化一个零矩阵 items，用于存储与这些用户相关的物品信息。
        items = torch.zeros([users.shape[0], args.n_items])
        #遍历这些用户，填充 items 矩阵以表示用户与物品之间的交互
        for j, user in enumerate(users):
            #将 items 矩阵中相应的位置设置为 1，表示该用户与特定物品之间的交互。
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)
        #计算余弦相似度，此处的分子是用户间共同物品数，分母是各自交互物品数的乘积的平方根
        sc_simi = (user_user_cap / ((items.sum(dim=1) *
                                     items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[users, i + args.n_users] = sc_simi
    #遍历每个用户以计算与物品之间的相似度，这个过程与物品到用户的计算类似。
    for i in range(args.n_users):
        #找出与用户 i 相连的所有物品索引。
        items = edge_index[1, edge_index[0] == i]
        #初始化矩阵以存储与这些物品相关的用户信息。
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        sc_simi = (item_item_cap / ((users.sum(dim=1) *
                                     users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[items, i] = sc_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]
    #将边权重矩阵保存到文件中
    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_sc.pt')

    return edge_weight

#基于 Leicht-Holme-Newman (LHN) 相似度的边权重，这种相似度常用于社交网络分析。LHN 相似度是一种归一化的度量，它考虑了两个节点的共同邻居数量和各自的度
def co_ratio_deg_user_lhn(adj_sp_norm, edge_index, degree, args):
    #将稀疏邻接矩阵 adj_sp_norm 转换为稠密格式，并仅保留表示用户和物品之间交互的部分
    user_item_graph = adj_sp_norm.to_dense()[:args.n_users, args.n_users:].cpu()
    #二值化 user_item_graph，将所有非零值设置为 1，表示存在交互。
    user_item_graph[user_item_graph > 0] = 1

    item_user_graph = user_item_graph.t()
    #初始化一个边权重矩阵，用于存储计算后的相似度值。
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))
#计算物品与用户之间的 LHN 相似度
    for i in range(args.n_items):
        #构建一个矩阵，表示这些用户之间的交互物品
        users = user_item_graph[:, i].nonzero().squeeze(-1)
        items = user_item_graph[users]
        #使用矩阵乘法计算用户间共同交互物品的数量
        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)
        #计算 LHN 相似度，即共同邻居数量除以各自邻居数量的乘积
        lhn_simi = (user_user_cap / ((items.sum(dim=1) *
                                      items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
        #将相似度值赋给边权重矩阵中相应的位置
        edge_weight[users, i + args.n_users] = lhn_simi

    for i in range(args.n_users):
        items = user_item_graph[i, :].nonzero().squeeze(-1)

        users = user_item_graph[:, items].t()
        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        lhn_simi = (item_item_cap / ((users.sum(dim=1) *
                                      users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[items + args.n_users, i] = lhn_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_lhn.pt')

    return edge_weight


def co_ratio_deg_user_lhn_sp(adj_sp_norm, edge_index, degree, args):
    edge_weight = torch.zeros(
        (args.n_users + args.n_items, args.n_users + args.n_items))

    for i in range(args.n_items):
        users = edge_index[0, edge_index[1] == i + args.n_users]
        items = torch.zeros([users.shape[0], args.n_items])

        for j, user in enumerate(users):
            items[j, torch.tensor(args.user_dict['train_user_set']
                                  [user.item()]) - args.n_users] = 1

        user_user_cap = torch.matmul(items, items.t())
        user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

        lhn_simi = (user_user_cap / ((items.sum(dim=1) *
                                      items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[users, i + args.n_users] = lhn_simi

    for i in range(args.n_users):
        items = edge_index[1, edge_index[0] == i]
        users = torch.zeros([items.shape[0], args.n_users])

        for j, item in enumerate(items):
            users[j, torch.tensor(
                args.user_dict['train_item_set'][item.item()])] = 1

        item_item_cap = torch.matmul(users, users.t())
        item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

        lhn_simi = (item_item_cap / ((users.sum(dim=1) *
                                      users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)

        edge_weight[items, i] = lhn_simi

    # print(edge_weight.nonzero().shape)
    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    torch.save(edge_weight, os.getcwd() + '/data/' +
               args.dataset + '/co_ratio_edge_weight_lhn.pt')

    return edge_weight

#用于可视化边权重与节点度数的关系。它创建了两个散点图，展示了边权重如何随着节点的度数变化。
def visual_edge_weight(edge_weight, edge_index, degree, dataname, type):
    deg_1 = degree[edge_index[0, :]]
    deg_2 = degree[edge_index[1, :]]

    plt.figure()
    #在散点图中以红色表示头节点的度数与边权重的关系
    plt.scatter(deg_1.tolist(), edge_weight.tolist(), s=1, c='red', label=type)
    #以绿色表示头节点的度数与 GCN（图卷积网络）中的归一化边权重（1/√deg_1 * 1/√deg_2）的关系。
    plt.scatter(deg_1.tolist(), 1 / deg_1**0.5 * 1 /
                deg_2**0.5, s=1, c='green', label='GCN', alpha=0.01)

    plt.xlabel('Degree of the head node', fontsize=15, fontweight='bold')
    plt.ylabel('Edge weight', fontsize=15, fontweight='bold')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + dataname +
                '/edge_weight' + type + '_head' + '.pdf', dpi=50, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(deg_2.tolist(), edge_weight.tolist(), s=1, c='red', label=type)
    plt.scatter(deg_2.tolist(), 1 / deg_1**0.5 * 1 /
                deg_2**0.5, s=1, c='green', label='GCN', alpha=0.01)

    plt.xlabel('Degree of the tail node', fontsize=15, fontweight='bold')
    plt.ylabel('Edge weight', fontsize=15, fontweight='bold')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + dataname +
                '/edge_weight' + type + '_tail' + '.pdf', dpi=50, bbox_inches='tight')
    plt.close()


def visual_node_rank(user_embs_aggr1, user_embs_aggr2, item_embs_aggr1, item_embs_aggr2, user_dict, topk, degree, n_users):
    #读取文件的所有行，然后选取第6行（假设数据从这里开始），因为索引是从0开始的
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    #计算两组用户和物品嵌入的点积，得到每组的评分矩阵。
    rating1 = torch.matmul(user_embs_aggr1, item_embs_aggr1.t())
    rating2 = torch.matmul(user_embs_aggr2, item_embs_aggr2.t())

    deg1, deg2 = [], []
    for i in range(n_users):
        color = []
        #计算两组用户和物品嵌入的点积，得到每组的评分矩阵。
        clicked_items = train_user_set[i] - n_users
        test_groundTruth_items = test_user_set[i] - n_users
        #将用户已点击的物品的评分设置为一个很低的值，以便在推荐时排除这些物品。
        rating1[i, clicked_items] = -(1 << 10)
        rating2[i, clicked_items] = -(1 << 10)
        #对每个用户，找出两组评分矩阵中评分最高的 topk 个物品及其索引
        rating_K1, idx_K1 = torch.topk(rating1[i], k=topk)
        rating_K2, idx_K2 = torch.topk(rating2[i], k=topk)

        rating_K1, idx_K1 = rating_K1.cpu(), idx_K1.cpu()
        rating_K2, idx_K2 = rating_K2.cpu(), idx_K2.cpu()

        for idx in idx_K1:
            if idx.item() in test_groundTruth_items and idx.item() not in idx_K2:
                deg1.append(degree[idx + n_users].item())
                deg2.append(degree[i])

        if i > 1000:
            break

    plt.figure()
    plt.scatter(deg1, deg2)
    plt.xlabel('item degree')
    plt.ylabel('user degree')
    plt.savefig('degree.pdf')
    plt.close()

#这个 plot_time 函数旨在从一个日志文件中读取时间、迭代次数（epoch）、召回率（recall）和 NDCG 指标，并绘制模型性能（如召回率）随训练周期变化的曲线。
def plot_time(dataset, model):
    with open('result_' + model + '_' + dataset + '_neg0.txt', 'r') as f:
        data = list(f.readlines())[5]

    time, epoch, recall, ndcg = [], [], [], []
    for i in range(len(data)):
        if not i % 2:
            time.append(int(data[i][2:9]))
            epoch.append(int(data[i][14:15]))
            recall.append(int(data[i][58:64]))
            ndcg.append(int(data[i][104:110]))

    plt.figure()
    plt.plot(epoch, recall)
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.savefig('result/' + dataset + '/' + model +
                '/performance_curve.pdf', dpi=100)
    plt.close()
