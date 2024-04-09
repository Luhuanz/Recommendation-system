import os

from parse import parse_args
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time

import numpy as np
import copy
import pickle

from utils import *
from evaluation import *
from model import *
from dataprocess import *


def run(model, optimizer, train_cf, clicked_set, user_dict, adj, args):
    #初始化最佳召回率 test_recall_best 为负无穷大，用于后续比较和更新召回率的最大值
    #early_stop_count 为 0，用于跟踪模型性能没有改进的连续迭代次数
    test_recall_best, early_stop_count = -float('inf'), 0
#调用 normalize_edge 函数来归一化邻接矩阵 adj。此函数返回归一化的邻接矩阵 adj_sp_norm 和各节点的度数 deg
    adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
    #从归一化的稀疏邻接矩阵 adj_sp_norm 中提取边的索引 edge_index 和边的权重 edge_weight。
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()
#将归一化的邻接矩阵 adj_sp_norm，边的索引 edge_index，边的权重 edge_weight，以及节点度数 deg 转移到模型指定的设备（如 GPU）上
    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)
#解包边的索引 edge_index，其中 row 代表边的起始节点索引，col 代表边的终止节点索引。
    row, col = edge_index
    args.user_dict = user_dict

    if args.model == 'CAGCN':
        if args.type == 'jc':
            if args.dataset in ['amazon']:
                cal_trend = co_ratio_deg_user_jacard_sp
            else:
                cal_trend = co_ratio_deg_user_jacard
        elif args.type == 'co':
            if args.dataset in ['amazon']:
                cal_trend = co_ratio_deg_user_common_sp
            else:
                cal_trend = co_ratio_deg_user_common
        elif args.type == 'lhn':
            if args.dataset in ['amazon']:
                cal_trend = co_ratio_deg_user_lhn_sp
            else:
                cal_trend = co_ratio_deg_user_lhn
        elif args.type == 'sc':
            if args.dataset in ['amazon']:
                cal_trend = co_ratio_deg_user_sc_sp
            else:
                cal_trend = co_ratio_deg_user_sc

        path = os.getcwd() + '/data/' + args.dataset + \
            '/co_ratio_edge_weight_' + args.type + '.pt'

        if os.path.exists(path):
            trend = torch.load(path)
        else:
#如果指定路径的文件不存在，则计算 trend。这里通过调用 cal_trend 函数进行计算，
# 并记录计算时间。cal_trend 函数可能是用来根据 adj_sp_norm、edge_index、deg 和其他参数计算图的趋势或特征。
            print(args.dataset, 'calculate_CIR', 'count_time...')
            start = time.time()
            trend = cal_trend(adj_sp_norm, edge_index, deg, args)
            print('Preprocession', time.time() - start)
        #计算当前趋势的归一化：
        norm = scatter_add(edge_weight, col, dim=0,dim_size=args.n_users + args.n_items)[col]
        norm_now = scatter_add(trend, col, dim=0, dim_size=args.n_users + args.n_items)[col]
#根据计算出的归一化因子更新 trend 值，使用了一个称为 trend_coeff 的系数来调节 trend 的影响。
        trend = args.trend_coeff * trend / norm_now * norm

        model.trend = (trend).to(args.device)
        args.model = args.model + '-' + args.type

    losses, recall, ndcg, precision, hit_ratio, F1 = defaultdict(list), defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    start = time.time()

    for epoch in range(args.epochs):
        neg_cf = neg_sample_before_epoch(
            train_cf, clicked_set, args)

        dataset = Dataset(
            users=train_cf[:, 0], pos_items=train_cf[:, 1], neg_items=neg_cf, args=args)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=dataset.collate_batch, pin_memory=args.pin_memory)  # organzie the dataloader based on re-sampled negative pairs

        """training"""
        model.train()
        loss = 0

        for i, batch in enumerate(dataloader):
            batch = batch_to_gpu(batch, args.device)

            user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0 = model(
                batch)

            bpr_loss = cal_bpr_loss(
                user_embs, pos_item_embs, neg_item_embs)

            # l2 regularization
            l2_loss = cal_l2_loss(
                user_embs0, pos_item_embs0, neg_item_embs0, user_embs0.shape[0])

            batch_loss = bpr_loss + args.l2 * l2_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()

        #******************evaluation****************
        if not epoch % 5:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Training_loss",
                               "Recall", "NDCG", "Precision", "Hit_ratio", "F1"]

            user_embs, item_embs = model.generate()
            test_res = test(user_embs, item_embs, user_dict, args)
            res.add_row(
                [format(time.time() - start, '.4f'), epoch, format(loss / (i + 1), '.4f'), test_res['Recall'], test_res['NDCG'],
                 test_res['Precision'], test_res['Hit_ratio'], test_res['F1']])

            print(res)

            for k in args.topks:
                recall[k].append(test_res['Recall'])
                ndcg[k].append(test_res['NDCG'])
                precision[k].append(test_res['Precision'])
                hit_ratio[k].append(test_res['Hit_ratio'])
                F1[k].append(test_res['F1'])
                losses[k].append(loss / (i + 1))

            # *********************************************************
            if test_res['Recall'][3] > test_recall_best:
                test_recall_best = test_res['Recall'][3]
                early_stop_count = 0

                if args.save:
                    torch.save(model.state_dict(), os.getcwd() +
                               '/trained_model/' + args.dataset + '/' + args.model + str(args.neg_in_val_test) + '.pkl')


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)

    """build dataset"""
    train_cf, test_cf, user_dict, args.n_users, args.n_items, clicked_set, adj = load_data(
        args)

    print(args.n_users, args.n_items, train_cf.shape[0] + test_cf.shape[0],
          (train_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users))

    if(args.neg_in_val_test == 1):  # whether negative samples from validation and test sets
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)
    elif args.model == 'NGCF':
        model = NGCF(args).to(args.device)
    elif args.model == 'MF':
        model = MF(args).to(args.device)
    elif args.model == 'CAGCN':
        model = CAGCN(args).to(args.device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run(model, optimizer, train_cf, clicked_set, user_dict, adj, args)
