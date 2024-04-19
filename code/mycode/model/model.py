# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv, ChebConv


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, normalize=True)
        self.conv2 = SAGEConv(h_feats, out_feats, normalize=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        x = self.conv2(x, edge_index,edge_weight)

        return x


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        x = self.conv2(x, edge_index,edge_weight)

        return x


class ChebNet(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_feats, h_feats, K=2)
        self.conv2 = ChebConv(h_feats, out_feats, K=2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        x = self.conv2(x, edge_index,edge_weight)
        return x




class SGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SGCN, self).__init__()
        self.conv = SGConv(in_feats, out_feats, K=2, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = self.conv(x, edge_index,edge_weight)

        return x
