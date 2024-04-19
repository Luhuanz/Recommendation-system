from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os.path as osp
from torch_geometric.datasets import Planetoid, WebKB, Actor, Reddit, CoraFull, Coauthor  #  用于加载特定图数据集的类，而不是直接导入数据集本身。这些类提供了一种机制，通过它们你可以下载（如果本地没有的话）、加载和处理相应的图数据集。
import torch_geometric.transforms as T   # 提供了对图数据进行预处理和转换的工具
import torch
import numpy as np
# homo
def get_dataset(name, normalize_features):
    if(name in ['Cora', 'Citeseer', 'Pubmed']): # 这些都是同构图数据集，通常用于节点分类任务。这些图中的节点代表科学文献，边代表文献之间的引用关系。
        load = Planetoid
    elif(name in ['Wisconsin', 'Cornell', 'Texas']):
        load = WebKB
    elif(name in ['Actor']):
        load = Actor
    elif(name in ['ogbn-arxiv', 'ogbn-products']):
        load = PygNodePropPredDataset
    elif(name in ['Reddit']):  #加载Reddit数据集，这是一个大型的社交网络数据集，节点通常代表用户或帖子，边表示用户间的交互或帖子的引用关系。 同构图数据集
        load = Reddit
    elif(name in ['CoraFull']):
        load = CoraFull
    elif(name in ['CS', 'Physics']): #它们是同构图数据集，用于节点分类任务。这些图中的节点代表科学文献，边代表文献之间的引用关系。
        load = Coauthor

    dataset = load_dataset(name=name, load=load, normalize_features=normalize_features,transform=None)
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
    # print(path)
    # exit()
    dataset = load(root=path,name=name) # 生成类对象 其实就是输入该类数据然后拿到对应数据集的数据
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def index_to_mask(index, size):
    #创建一个指定大小的全0布尔张量，设备与index相同
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
def pre_process(dataset, name):
    #dataset :Cora() # 大多是单图
    "2708个节点，每个节点有1433个特征。"
    "10556边"
    data = dataset[0]   #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    print(data)
    exit()
    if(name in ['ogbn-arxiv', 'ogbn-products']):
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