# -*- coding:utf-8 -*-
import copy
import random
import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from utils import *
from model import *
from torch_scatter import scatter_max, scatter_add
path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"
device='cuda' if torch.cuda.is_available() else 'cpu'
filename='model_classification.txt'
file_exists=os.path.exists(filename)
@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
   # pred=out[data.test_mask].argmax(dim=1)
  #  acc = int(pred.eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
    model.train()

    return loss.item(), acc


def train(model, data, args):
    data = data.to(device)
    model.to(device)
    acc = np.zeros(args.runs)
    for count in tqdm(range(args.runs), unit=args.dataset):
        seed_everything(args.seed + count)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.005)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        min_val_loss = np.Inf
        final_test_acc=-np.Inf
        best_model = None

        for epoch in range(200):
            model.train()
            out = model(data)
            optimizer.zero_grad()
            loss = loss_function(out[data.train_mask], data.y[data.train_mask].to(device))
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, test_acc = test(model, data)
            if val_loss < min_val_loss and epoch > 5:
                min_val_loss = val_loss
                final_test_acc = test_acc
                best_model = copy.deepcopy(model)

        acc[count] = final_test_acc
        tqdm.write(f'Run {count:03d}: Val Loss: {min_val_loss:.4f}, Test Acc: {final_test_acc:.4f}')

    with open(filename, 'a+') as f:
        if not file_exists:
            f.write('Model\tDataset\tEdge_Weight\tnormalize_features\tMean\tStd\n')
        f.write(f'{args.model}\t{args.dataset}\t{args.edge_weight}\t{args.normalize_features}\t{np.mean(acc):.4f}\t{np.std(acc):.4f}\n')

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11131)
    parser.add_argument('--cir_coeff', type=float, default=0.5) # circular coefficient
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--edge_weight', type=bool, default=False)
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument("--runs",type=int,default=50)
    args = parser.parse_args()
    ## name: CiteSeer Cora  PubMed..
    # import the dataset
    dataset = get_dataset(name=args.dataset, normalize_features=args.normalize_features)  # Cora() -》类的实例化
    # print(dataset)
    # exit()
    # data, num_in_feats, num_out_feats=load_data(path,name=args.dataset) # data is a PyG data object  num_out_feats is class
    data = pre_process(dataset, args.dataset) #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    num_in_feats, num_out_feats=dataset.num_features, dataset.num_classes
    if args.edge_weight:
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0)) # 每个都加环    idea  随机加环？

        # print(data.edge_index.shape) #torch.Size([2, 13264])
        # calculate the degree normalize term
        row, col = data.edge_index # 解构edge中的边
        # print(data.x.shape) #torch.Size([2708, 1433])
        # print(data.y.shape) #torch.Size([2708])
        # exit()
        deg = degree(col, data.x.size(0), dtype=data.x.dtype)  #只关注入度
        deg_inv_sqrt = deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  #这种归一化，可以改善训练过程的稳定性，并提高模型在图数据上的性能。
        new_edge_weight = cal_sc(data.edge_index, data.x, args)  # **** create torch.Size([13264])

        norm = scatter_add(edge_weight, col, dim=0,  #（例每个节点从其所有入边接收到的信息量如权重总和）
                           dim_size=data.x.size(0))[col]  #对每个节点的所有入边的权重进行加和
        # print(norm.shape) #torch.Size([13264])
        # exit()
        norm_now = scatter_add(new_edge_weight, col, dim=0,
                               dim_size=data.x.size(0))[col]
        # print(norm_now.shape)
        # exit()
        new_edge_weight = args.cir_coeff * \
            new_edge_weight / norm_now + edge_weight   # args.cir_coeff 平衡不同Norm贡献
        data.edge_weight = new_edge_weight
        # print(data.edge_weight.shape)
        # exit()#torch.Size([13264])

    num_edges = data.edge_index.size(1)  # 获取边的数量
    unit_edge_weights = torch.ones(num_edges, dtype=torch.float)  # 创建单位权重
    data.edge_weight=unit_edge_weights     #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_weight=[10556])
    if args.model == 'GCN':
        model = GCN(num_in_feats, args.hidden, num_out_feats).to(device)
    elif args.model == 'GraphSAGE':
        model = GraphSAGE(num_in_feats, args.hidden, num_out_feats).to(device)
    elif args.model == 'GAT':
        model = GAT(num_in_feats, args.hidden, num_out_feats).to(device)
    # elif args.model == 'SGC':
    #     model = SGC(num_in_feats, num_out_feats).to(device)
    elif args.model == 'SGCN':
        model = SGCN(num_in_feats, num_out_feats).to(device)
    else:
        raise NotImplementedError('model not implemented')

    acc = train(model, data,args)
    print(f'test acc:',np.mean(acc))