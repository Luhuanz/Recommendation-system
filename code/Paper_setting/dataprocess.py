import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch.utils.data import Dataset as BaseDataset
import os
from torch_geometric.utils import add_remaining_self_loops, degree

#每一行表示一个交互，其中第一列是用户ID，第二列是项目ID cf:Collaborative Filtering
def read_cf_list(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_by_user(file_name):
    #初始化inter_mat为一个空列表，用于存储用户和物品之间的交互数据
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        #将每行数据按空格分割，并转换为整数列表 inters
        inters = [int(i) for i in tmps.split(" ")]
        #将 inters 列表的第一个元素指定为用户 ID u_id，剩余的元素作为该用户的正样本物品 ID 列表 pos_ids
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            #将用户 ID 和物品 ID 作为一对添加到 inter_mat 列表中，形成交互对
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

#处理训练和测试数据，用于推荐系统中的数据预处理步骤。它计算用户和物品的数量，并为每个用户和物品构建了交互列表。
def process(train_data, test_data):
    #计算用户数量，即找出训练和测试数据中的最大用户 ID，并加 1（因为 ID 从 0 开始）
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1 #29858
    #计算物品数量，即找出训练和测试数据中的最大物品 ID，并加 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1 #40981
#初始化三个字典 train_user_set、test_user_set 和 train_item_set，用于存储训练集中用户和物品的交互信息和测试集中用户的交互信息。
    train_user_set, test_user_set, train_item_set = defaultdict(
        list), defaultdict(list), defaultdict(list)
#将物品 ID 增加用户数量，这是为了在后续处理中区分用户和物品的 ID
    train_data[:, 1] += n_users #(810128, 2)
    test_data[:, 1] += n_users
#构建训练集用户到物品的交互集合和物品到用户的交互集合
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
#构建测试集用户到物品的交互集合
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

    return n_users, n_items, train_user_set, test_user_set, train_item_set


def process_adj(data_cf, n_users, n_items):
    #创建交互数据 data_cf 的副本，以避免修改原始数据
    cf = data_cf.copy()
    #这一行实际上没有改变 cf 的内容。原本可能是想将物品 ID 映射到新的范围，但这里只是简单地赋值给自己。
    cf[:, 1] = cf[:, 1]  # [0, n_items) -> [n_users, n_users+n_items)
    #再次复制 cf，用于创建用户到物品和物品到用户的双向关系
    cf_ = cf.copy()
    #交换 cf_的列，使得原来表示用户到物品的关系变成物品到用户的关系，实现双向关联
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
   #将原始的用户 - 物品关系和反向的物品 - 用户关系合并，构成完整的邻接信息
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    return torch.LongTensor(cf_).t()

#计算给定邻接矩阵的归一化拉普拉斯矩阵，具体是使用对称归一化的方法。
def _bi_norm_lap(adj):
    # D^{-1/2}AD^{-1/2}
    #邻接矩阵 adj 的每一行之和，得到节点的度数（或邻接点的总数）。这里的 sum(1) 表示按行求和。
    rowsum = np.array(adj.sum(1))
    #计算度数矩阵的逆平方根，并将其展平。这里，np.power(rowsum, -0.5) 计算每个度数的 -0.5 次幂，实现了矩阵 D 的逆平方根计算
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    #将结果转换为 COOrdinate 格式的稀疏矩阵，并返回
    return bi_lap.tocoo()

#计算给定邻接矩阵的单侧归一化的拉普拉斯矩阵。这种归一化方式采用度矩阵的逆，而不是逆平方根。
def _si_norm_lap(adj):
    # D^{-1}A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()


def load_data(args):
    print('reading train and test user-item set ...')
    #使用 read_cf_list 函数从文件中读取训练和测试数据。这些数据文件的路径由当前工作目录、数据集名称和分割编号组成。
    train_cf = read_cf_list(os.getcwd() + '/data/' +args.dataset + '/train' + str(args.split) + '.txt')
    test_cf = read_cf_list(os.getcwd() + '/data/' + args.dataset + '/test' + str(args.split) + '.txt')
    #该函数处理读入的训练和测试数据，返回用户总数、物品总数，以及训练和测试集中用户和物品的交互集合。
    n_users, n_items, train_user_set, test_user_set, train_item_set = process(train_cf, test_cf)

    print('building the adj mat ...')
    #使用 process_adj 函数处理训练数据 train_cf，构建用于图模型的邻接矩阵
    adj = process_adj(train_cf, n_users, n_items)

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set,
    }

    clicked_set = defaultdict(list)
    for key in user_dict:
        for user in user_dict[key]:
            clicked_set[user].extend(user_dict[key][user])

    print('loading over ...')
    return train_cf, test_cf, user_dict, n_users, n_items, clicked_set, adj

#该类继承自 BaseDataset，并且实现了一些必要的方法来支持数据加载和批处理。下面是对代码的逐行解释：
class Dataset(BaseDataset):
    def __init__(self, users, pos_items, neg_items, args, link_ratios=None):
        #接收用户、正样本物品、负样本物品和参数 args，以及可选的链接比例 link_ratios（这里被注释掉了
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
        # self.link_ratios = link_ratios
        self.args = args
    #根据索引 index，构造并返回一个字典，包含了指定索引处的用户、正样本物品和负样本物品。
    def _get_feed_dict(self, index):

        # print(self.weight.shape)
        feed_dict = {
            'users': self.users[index],
            'pos_items': self.pos_items[index],
            'neg_items': self.neg_items[index],
            # 'link_ratios': self.link_ratios[index]
        }

        return feed_dict
#返回用户列表的长度，即数据集中的样本总数
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self._get_feed_dict(index)
#用于批处理时将多个样本组合成一个批次的数据。
    def collate_batch(self, feed_dicts):
        # feed_dicts: [dict1, dict2, ...]
# 创建一个新的字典 feed_dict，其中包含所有样本的用户、正样本物品和负样本物品的集合
        feed_dict = dict()

        feed_dict['users'] = torch.LongTensor([d['users'] for d in feed_dicts])
        feed_dict['pos_items'] = torch.LongTensor(
            [d['pos_items'] for d in feed_dicts])

        feed_dict['neg_items'] = torch.LongTensor(
            np.stack([d['neg_items'] for d in feed_dicts]))
        # feed_dict['link_ratios'] = torch.LongTensor(
        #     np.stack([d['link_ratios'] for d in feed_dicts]))

        feed_dict['idx'] = torch.cat(
            [feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'].view(-1)])

        return feed_dict

#这个normalize_edge函数用于对图的边进行归一化处理，类似于计算对称归一化的拉普拉斯矩阵中的一步。
def normalize_edge(edge_index, n_users, n_items):
    # edge_index, _ = add_remaining_self_loops(edge_index)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#它表示归一化的邻接矩阵，以及节点的度数。这个稀疏张量使用提供的边索引和计算出的边权重创建，其形状为 (n_users + n_items, n_users + n_items)，表示整个用户 - 物品图。
    return torch.sparse.FloatTensor(edge_index, edge_weight, (n_users + n_items, n_users + n_items)), deg

#从文本文件中读取训练和测试交互数据，并将它们转换为整数矩阵形式，最后再写回到文件中
def transform_data2(dataset):
    train_inter_mat, test_inter_mat = list(), list()

    # training
    lines = open(os.getcwd() + '/data/' + dataset +'/train1.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip().split(",")
        train_inter_mat.append([int(tmps[0]), int(tmps[1])])

    lines = open(os.getcwd() + '/data/' + dataset +'/test1.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip().split(",")
        test_inter_mat.append([int(tmps[0]), int(tmps[1])])
##将处理后的训练和测试数据（现在是整数形式的 numpy 数组）写回到原始文件路径。fmt='%i' 指定保存数据时使用整数格式。
    np.savetxt(os.getcwd() + '/data/' + dataset +'/train1.txt', np.array(train_inter_mat), fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +'/test1.txt', np.array(test_inter_mat), fmt='%i')

#目的是从原始文件中读取用户 - 物品交互数据，并转换为整数矩阵形式，再将这些数据保存到新的文件中。
def transform_data1(dataset):
    train_inter_mat, test_inter_mat = list(), list()

    # training
    lines = open(os.getcwd() + '/data/' + dataset +'/train.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]

        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            train_inter_mat.append([u_id, i_id])

    # testing
    lines = open(os.getcwd() + '/data/' + dataset +'/test.txt', 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]

        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            test_inter_mat.append([u_id, i_id])

    np.savetxt(os.getcwd() + '/data/' + dataset +'/train1.txt', np.array(train_inter_mat), fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +'/test1.txt', np.array(test_inter_mat), fmt='%i')


def preprocess_edges(file, dataset):
    #从指定文件加载边列表。文件中的每一行应该代表一个边，通常包含两个整数表示用户和物品ID。
    edge_list = np.loadtxt(file, dtype=int)
    #计算用户集中每个节点的度数（边的数量）。这里使用PyTorch的degree函数，假设edge_list的第一列包含用户ID。
    deg = degree(torch.tensor(edge_list[:, 0]),num_nodes=max(edge_list[:, 0]) + 1)
    #通过找出每列的最大 ID 并加一（假设 ID 从零开始），确定用户和物品的总数。
    num_user, num_item = max(edge_list[:, 0]) + 1, max(edge_list[:, 1]) + 1
    #在存在交互的地方填充邻接矩阵，将对应的单元格设置为 1
    adj = torch.zeros((num_user, num_item))
    adj[edge_list[:, 0], edge_list[:, 1]] = 1
    #过滤出度数大于 10 的用户，将邻接矩阵减少到只包含活跃用户
    adj = adj[deg > 10]
    #将邻接矩阵转换回边列表格式，其中每行包含非零元素的索引（用户 - 物品对）
    edge_list = adj.nonzero().numpy()
    idx = np.arange(len(edge_list))
    #随机选择 80% 的索引作为训练数据
    train_idx = np.random.choice(idx, size=int(edge_list.shape[0] * 0.8), replace=False)
    #初始化一个用于过滤的布尔数组。
    filtering = np.full(edge_list.shape[0], False, dtype=bool)
    filtering[train_idx] = True

    train_edge_list = edge_list[filtering]
    test_edge_list = edge_list[~filtering]
    #将过滤后的训练和测试边列表分别保存到新的文件中。这个过程帮助确保训练和测试数据的分割是随机和合理的，便于后续的机器学习和数据分析任务。
    np.savetxt(os.getcwd() + '/data/' + dataset +'/train1.txt', train_edge_list, fmt='%i')
    np.savetxt(os.getcwd() + '/data/' + dataset +'/test1.txt', test_edge_list, fmt='%i')


# preprocess_edges('./data/worldnews/worldnews_edges_2year.txt', 'worldnews')
