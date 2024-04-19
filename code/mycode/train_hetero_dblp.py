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
def test(model, data,args):
    model.eval()
    out = model(data)
    y = args.data_map['y'].to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[args.data_map['val_mask']], y[args.data_map['val_mask']])
    _, pred = out.max(dim=1)
    correct = int(pred[args.data_map['test_mask']].eq(y[args.data_map['test_mask']]).sum().item())
    acc = correct / int(args.data_map['test_mask'].sum())
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        min_val_loss = np.Inf
        final_test_acc=-np.Inf
        best_model = None
        y = args.data_map['y'].to(device)
        for epoch in range(200):
            model.train()
            out = model(data)
            optimizer.zero_grad()
            loss = loss_function(out[args.data_map['train_mask']], y[args.data_map['train_mask']].to(device))
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, test_acc = test(model, data,args)
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
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cir_coeff', type=float, default=0.5) # circular coefficient
    parser.add_argument('--model', type=str, default='HomoFeatureRGCN')
    parser.add_argument('--edge_weight', type=bool, default=False)
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument("--runs",type=int,default=50)
    parser.add_argument('--in_feats', type=int, default=128)
    parser.add_argument('--data_map', type=dict, default={})
    args = parser.parse_args()
    # import the dataset
    dataset = get_hetero_dataset(name=args.dataset, normalize_features=args.normalize_features)  #   -》类的实例化
    # 打印图的一些基本信息
    # print("",dataset.num_classes)
    # exit()
    data, args.data_map = pre_hetero_process(dataset, args)
    num_out_feats=args.data_map['num_classes']
    # num_in_feats, num_out_feats=dataset.num_features, dataset.num_classes
    # num_edges = data.edge_index.size(1)  # 获取边的数量
    # unit_edge_weights = torch.ones(num_edges, dtype=torch.float)  # 创建单位权重
    # data.edge_weight=unit_edge_weights     #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_weight=[10556])
    if args.model == 'HomoFeatureRGCN':
        model = HomoFeatureRGCN(args.in_feats, args.hidden, num_out_feats,args.data_map).to(device)
    elif args.model == 'HeteroFeatureRGCN':
        model = HeteroFeatureRGCN(args.in_feats, args.hidden, num_out_feats,args.data_map).to(device)
    else:
        raise NotImplementedError('model not implemented')

    acc = train(model, data,args)
    print(f'test acc:',np.mean(acc))