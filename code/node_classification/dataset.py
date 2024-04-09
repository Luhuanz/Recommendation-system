import os.path as osp
from torch_geometric.datasets import Planetoid, WebKB, Actor, Reddit, CoraFull, Coauthor  #  用于加载特定图数据集的类，而不是直接导入数据集本身。这些类提供了一种机制，通过它们你可以下载（如果本地没有的话）、加载和处理相应的图数据集。
import torch_geometric.transforms as T   # 提供了对图数据进行预处理和转换的工具
import torch
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, degree #给图中的每个节点添加自环，这在某些图神经网络模型中是需要的，以确保每个节点在信息传递过程中也考虑自己的特征

def index_to_mask(index, size):
    #创建一个指定大小的全0布尔张量，设备与index相同
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_dataset(name, normalize_features=False, transform=None):
    if(name in ['Cora', 'Citeseer', 'Pubmed']):
        load = Planetoid
    elif(name in ['Wisconsin', 'Cornell', 'Texas']):
        load = WebKB
    elif(name in ['Actor']):
        load = Actor
    elif(name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-proteins']):
        load = PygNodePropPredDataset
    elif(name in ['Reddit']):  #加载Reddit数据集，这是一个大型的社交网络数据集，用于社区检测和图分类任务。
        load = Reddit
    elif(name in ['CoraFull']):
        load = CoraFull
    elif(name in ['CS', 'Physics']):
        load = Coauthor

    dataset = load_dataset(
        name, load=load, normalize_features=normalize_features)
    # print(dataset)
    # exit()
    return dataset


def load_dataset(name, load, normalize_features=False, transform=None):
    '''
     name: 数据集的名称。
    load: 用于加载数据集的类。
    normalize_features: 布尔值，指示是否应该归一化节点特征。
    transform: 应用于数据集的额外变换。
    dataset = load(root=path, name=name)  # 用于加载数据集的类
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = load(root=path, name=name) # 生成类对象 其实就是输入该类数据然后拿到对应数据集的数据
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def pre_process(dataset, name):
    #dataset :Cora() # 大多是单图
    "2708个节点，每个节点有1433个特征。"
    "10556边"
    data = dataset[0]   #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    if(name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']):
        #特征归一化：对节点特征x进行行归一化处理，确保每个节点的特征向量和为1。
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        data.edge_index = torch.stack([torch.cat([data.edge_index[0], data.edge_index[1]]), torch.cat(
            [data.edge_index[1], data.edge_index[0]])])
        #边索引扩展：生成无向图的边索引。由于OGB数据集默认可能是有向的
        split_idx = dataset.get_idx_split()
       #分割索引转换为掩码
        train_mask, val_mask, test_mask = index_to_mask(split_idx['train'], data.x.size(0)), index_to_mask(
            split_idx['valid'], data.x.size(0)), index_to_mask(split_idx['test'], data.x.size(0))
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
      #调整标签维度：如果标签y的维度不是一维的，通过data.y = data.y.view(-1, )调整
        data.y = data.y.view(-1, )
    return data

#旨在为Planetoid类型的图数据集（如Cora、Citeseer和Pubmed）生成随机的数据划分。
def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

if __name__ == '__main__':
    # # 定义图的总节点数
    # num_nodes = 10
    #
    # # 指定训练集和测试集的节点索引
    # train_index = torch.tensor([0, 1, 2])
    # test_index = torch.tensor([3, 4])
    #
    # # 使用index_to_mask函数创建掩码
    # train_mask = index_to_mask(train_index, num_nodes)  #Train Mask: tensor([ True,  True,  True, False, False, False, False, False, False, False])
    # test_mask = index_to_mask(test_index, num_nodes) #Test Mask:  tensor([False, False, False,  True,  True, False, False, False, False, False])
    #
    # # 打印结果
    # print("Train Mask:", train_mask)
    # print("Test Mask: ", test_mask)
    # 加载并归一化Cora数据集的特征
    # dataset = load_dataset(name='Cora', load=Planetoid)  类
    # print(type(dataset))
    pass