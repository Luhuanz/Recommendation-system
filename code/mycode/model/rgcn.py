from torch_geometric.nn import RGCNConv
import torch
import torch.nn as nn
import copy

class BaseRGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, 64, num_relations=num_relations)
        self.conv2 = RGCNConv(64, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_weight))
        x = self.conv2(x, edge_index, edge_type, edge_weight)
        return x

class HomoFeatureRGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats,data_map):
        super(HomoFeatureRGCN, self).__init__()
        self.num_relations=data_map['num_relations']
        self.init_sizes=data_map['init_sizes']
        self.node_types=data_map['node_types']
        self.conv1 = RGCNConv(in_feats, h_feats,
                              num_relations= self.num_relations, num_bases=30)
        self.conv2 = RGCNConv(h_feats, out_feats,
                              num_relations= self.num_relations, num_bases=30)
        self.lins = torch.nn.ModuleList()
        for i in range(len( self.node_types)):
            lin = nn.Linear( self.init_sizes[i], in_feats)
            self.lins.append(lin)
#trans_dimensions 方法起到一个重要的作用，尤其是在处理异构图（含多种类型的节点和边）时。在这个方法中，我们对每种类型的节点特征进行线性变换，以便与模型的输入维度匹配。
    #trans_dimensions 方法通过对每种类型的节点应用一个线性变换（nn.Linear），将所有节点的特征投影到一个统一的维度空间。这样做可以让模型更容易处理来自不同类型的节点的信息
    def trans_dimensions(self, g):
        data = copy.deepcopy(g) #首先深拷贝原始图数据，以确保不改变原始输入数据
        for node_type, lin in zip(self.node_types, self.lins):  #对每种节点类型应用预定义的线性变换，更新节点的特征
            data[node_type].x = lin(data[node_type].x)  #将更新特征后的图数据返回，供模型进一步处理
        return data
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
    def forward(self, data):
        num_nodes=data['author'].x.shape[0]
        data = self.trans_dimensions(data) # 统一节点特征维度
        # print(data)
        homogeneous_data = data.to_homogeneous() #Data(node_type=[26128], x=[26128, 128], edge_index=[2, 239566], edge_type=[239566])
        # print(homogeneous_data)
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = x[:num_nodes]

        return x

class RGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats,data_map):
        super(RGCN, self).__init__()
        self.num_relations=data_map['num_relations']
        self.conv1 = RGCNConv(in_feats, h_feats,
                              num_relations= self.num_relations, num_bases=30)
        self.conv2 = RGCNConv(h_feats, out_feats,
                              num_relations= self.num_relations, num_bases=30)

    def forward(self,data):
        pass