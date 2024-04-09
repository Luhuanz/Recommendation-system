import argparse
from dataset import *
from learn import *
from model import *
from utils import *
from os import path
from tqdm import tqdm
import random
from torch import tensor
from torch_scatter import scatter_max, scatter_add


def run(data, args):
    #args.runs是要执行的总运行次数,指定了进度条的单位，这里表示每次迭代代表一次运行。
    pbar = tqdm(range(args.runs), unit='run')
    acc = np.zeros(args.runs)
    data = data.to(args.device)
    model = GCN(args).to(args.device)
    for count in pbar:
        seed_everything(args.seed + count) # 确保实验的可重复性, 评估模型的鲁棒性和泛化能力
        model.reset_parameters() #重新初始化模型中所有参数的权重,每次实验开始时模型的状态都是一致的，从而保证实验结果的可比性和可重复性。
        optimizer = torch.optim.Adam([
            dict(params=model.lin1.parameters(), weight_decay=args.wd1),
            dict(params=model.bias1, weight_decay=args.wd1),
            dict(params=model.lin2.parameters(), weight_decay=args.wd2),
            dict(params=model.bias2, weight_decay=args.wd2)], lr=args.lr)
        best_val_loss = float('inf')
        val_loss_history = []
        for epoch in range(0, args.epochs):
            loss = train(model, data, optimizer, args)
            evals = evaluate(model, data, args)
            if loss['val'] < best_val_loss:
                best_val_loss = loss['val']
                test_acc = evals['test']
                # torch.save(model.state_dict(), 'model.pkl')
            val_loss_history.append(loss['val'])
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])# 这些损失值将用于计算平均损失
                if loss['val'] > tmp.mean().item():
                    break
        acc[count] = test_acc
        print(test_acc)
    return acc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd1', type=float, default=0.0008)
    parser.add_argument('--wd2', type=float, default=0.0000)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.9)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1028)
    parser.add_argument('--cir_coeff', type=float, default=0.8)
    parser.add_argument('--model', type=str, default='cagcn')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # import the dataset
    dataset = get_dataset(args.dataset, args.normalize_features)  #Cora() -》类的实例化
    # print(dataset)
    # exit()
    data = pre_process(dataset, args.dataset)#Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    # print(data)
    # exit()
    args.num_features, args.num_classes = dataset.num_features, dataset.num_classes #1433  7 dymatic 不同domain论文
    # print(data.edge_index.shape) #torch.Size([2, 10556])
    # exit()
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
    # Namespace(dataset='Cora', runs=10, epochs=1000, lr=0.001, wd1=0.0008, wd2=0.0, early_stopping=100, hidden=64,
    #           dropout=0.9, normalize_features=True, seed=1028, cir_coeff=0.8, model='cagcn', device=device(type='cuda'),
    #           num_features=1433, num_classes=7)
    new_edge_weight = cal_sc(data.edge_index, data.x, args)  # **** create
    norm = scatter_add(edge_weight, col, dim=0,  #（例每个节点从其所有入边接收到的信息量如权重总和）
                       dim_size=data.x.size(0))[col]  #对每个节点的所有入边的权重进行加和
    # print(norm.shape) #torch.Size([13264])
    norm_now = scatter_add(new_edge_weight, col, dim=0,
                           dim_size=data.x.size(0))[col]
    #print(norm_now.shape) #torch.Size([13264])
    new_edge_weight = args.cir_coeff * \
        new_edge_weight / norm_now + edge_weight   # args.cir_coeff 平衡不同Norm贡献

    data.edge_weight = new_edge_weight

    acc = run(data, args)

    print(np.mean(acc), np.std(acc))

