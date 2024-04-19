from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os.path as osp
from .sub_graph_samping import k_hop_subgraph
import torch_geometric.transforms as T  # 提供了对图数据进行预处理和转换的工具
import torch
import numpy as np
from torch_geometric.datasets import DBLP
from torch_geometric.datasets import Entities



# hetero
def get_hetero_dataset(name, normalize_features):
    if (name in ['DBLP']):
        dataset = load_DBLP(name=name, normalize_features=normalize_features, transform=None)
    elif (name in ['AIFB', 'MUTAG', 'BGS', 'AM']): #：AIFB一个关于学术信息的知识图谱，通常用于关系预测和实体分类任务。BGS ：英国地质调查数据集，用于地质和地理实体的分类 AMAM：应用领域：类似于 AIFB，通常用于学术领域的知识图谱。
        # from dgl
        dataset = load_Entities(name="Entities",data=name, normalize_features=normalize_features, transform=None)
    elif (name in ['ogbn-mag', 'ogbn-proteins']):  #ogbn-mag是一个异构图数据集,ogbn-proteins：可以被视为异构图
        load = PygNodePropPredDataset
        dataset = load_ogb(name=ogbn, load=name, normalize_features=normalize_features, transform=None)

    return dataset


def load_DBLP(name,normalize_features=False, transform=None):
    '''
     name: 数据集的名称。
    load: 用于加载数据集的类。
    normalize_features: 布尔值，指示是否应该归一化节点特征。
    transform: 应用于数据集的额外变换。
    dataset = load(root=path, name=name)  # 用于加载数据集的类
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = DBLP(root=path)  # 生成类对象 其实就是输入该类数据然后拿到对应数据集的数据
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def load_ogb(name,data, normalize_features, transform=None):
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
    dataset = load(root=path, name=name)  # 生成类对象 其实就是输入该类数据然后拿到对应数据集的数据
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def load_Entities(name,data, normalize_features, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    # print(path)
    # exit()
    dataset = Entities(root=path, name=data)  # 生成类对象 其实就是输入该类数据然后拿到对应数据集的数据
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


def pre_hetero_process(dataset, args):
    #Data(edge_index=[2, 58086], edge_type=[58086], train_idx=[140], train_y=[140], test_idx=[36], test_y=[36], num_nodes=8285)
    data = dataset[0]
    res = {}
    if(args.dataset in ['DBLP']):
        num_classes = torch.max(data['author'].y).item() + 1 #pytorch node need+1
        data['conference'].x = torch.randn((data['conference'].num_nodes, 50)) #会议节点没有初始特征，为会议节点创建了一个随机特征矩阵，
        train_mask, val_mask, test_mask = data['author'].train_mask, data['author'].val_mask, data['author'].test_mask
        y = data['author'].y
        node_types, edge_types = data.metadata() #获取图中所有节点和边的类型 #node_types 是一个列表，包含图中不同节点的类型（如作者、论文、会议等
        # print(node_types) #['author', 'paper', 'term', 'conference']
        # print(edge_types) #[('author', 'to', 'paper'), ('paper', 'to', 'author'), ('paper', 'to', 'term'), ('paper', 'to', 'conference'), ('term', 'to', 'paper'), ('conference', 'to', 'paper')]
        num_nodes = data['author'].x.shape[0] #author number
        num_relations = len(edge_types)
        init_sizes = [data[x].x.shape[1] for x in node_types] #每种节点类型的特征维数
        # print(init_sizes) #[334, 4231, 50, 50] ['author', 'paper', 'term', 'conference']
        # print(data) heterogeneous graph
        # graph = data.to_homogeneous() #Data(node_type=[26128], edge_index=[2, 239566], edge_type=[239566])
        res['num_classes'] = num_classes
        res['train_mask'] = train_mask
        res['val_mask'] = val_mask
        res['test_mask'] = test_mask
        res['num_relations']=num_relations
        res['init_sizes']=init_sizes
        res['node_types']=node_types
        res['y']=y
        return data,res
    #Data(edge_index=[2, 58086], edge_type=[58086], train_idx=[140], train_y=[140], test_idx=[36], test_y=[36], num_nodes=8285)
    if (args.dataset in ['AIFB', 'MUTAG', 'BGS', 'AM']):
        node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
        node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, 2, data.edge_index, relabel_nodes=True)
        data.num_nodes = node_idx.size(0)
        data.edge_index = edge_index
        data.edge_type = data.edge_type[edge_mask]
        data.train_idx = mapping[:data.train_idx.size(0)]
        data.test_idx = mapping[data.train_idx.size(0):]   #subgraph
      #  print(data)#Data(edge_index=[2, 54024], edge_type=[54024], train_idx=[140], train_y=[140], test_idx=[36], test_y=[36], num_nodes=6909)
      # 表示图中的边索引 edge_type 表示每条边的类型。边的类型用于区分图中不同类型的连接，这是异构图的一个典型特征 tyge_type 来进行更复杂的图结构分析，比如使用专为处里异构图
        res['num_classes']=dataset.num_classes
        res['num_relations']=dataset.num_relations
    return data
    if (args.dataset in ['ogbn-proteins']):
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
#