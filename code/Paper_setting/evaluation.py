import numpy as np

from utils import *
#设置NumPy打印选项，使得浮点数打印时保留四位小数
np.set_printoptions(precision=4)


def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        #获取第i个测试样本的真实数据（ground truth）
        groundTrue = test_data[i]
        #获取第 i 个测试样本的预测数据。
        predictTopK = pred_data[i]
        #使用 map 函数和 lambda 表达式对 predictTopK 中的每个元素进行判断，看它是否存在于 groundTrue 中，结果是一个布尔值列表
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')

#用于计算在前 k 个推荐项中是否有正确推荐的情况，即“命中率”指标 返回的是每个测试样本在前 k 个推荐中是否有命中的二进制标识符
def Hit_at_k(r, k):
#对于二进制数组 r（其中的行代表测试样本，列代表是否正确预测），这行代码计算每个样本前 k 个推荐中正确预测的数量。通过沿着轴 1（即列）求和，我们得到每个样本的正确预测数量。
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)

#用于计算在前k个推荐项中的召回率（Recall）和精确率（Precision）
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    # print(right_pred, 2213123213)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}

#NDCGatK_r 函数计算在前k个推荐项中的归一化折损累积增益（Normalized Discounted Cumulative Gain, NDCG）。这是衡量推荐系统排序质量的常用指标
def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
#首先确认预测结果 r 和测试数据 test_data 的长度相同，确保每个用户都有对应的预测和实际结果。
    assert len(r) == len(test_data)
    #从预测结果中选取前 k 个推荐项
    pred_data = r[:, :k]
#初始化 test_matrix，用于存储测试数据中的真实反馈，其中每一行代表一个用户，每一列代表是否为真正感兴趣的项
    test_matrix = np.zeros((len(pred_data), k))
    #遍历测试数据 test_data，为每个用户的前 k 个项（或实际项数，取较小值）设置为 1
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
#计算理想情况下的累积增益（Ideal Discounted Cumulative Gain, IDCG）
    # print(max_r[0], pred_data[0]) 通过对 max_r 中的每一项乘以折损因子（基于位置的逆对数）并按行求和
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    #计算 NDCG 值，即每个用户的 DCG 除以 IDCG。
    ndcg = dcg / idcg
    # ndcg[np.isnan(ndcg)] = 0.

    return ndcg


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp
        # F1s[np.isnan(F1s)] = 0

        F1.append(sum(F1s))

    return {'Recall': np.array(recall),
            'Precision': np.array(pre),
            'NDCG': np.array(ndcg),
            'F1': np.array(F1),
            'Hit_ratio': np.array(hit_ratio)}


def test(user_embs, item_embs, user_dict, args):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []
        #使用 minibatch 函数生成小批量的测试用户
        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            #计算每批用户对所有物品的评分矩阵 其中每一行代表一个用户对所有物品的评分。
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())
    #取每个用户在训练集中交互过的物品。这里假设物品 ID 是从 n_users 开始编号的，所以从物品 ID 中减去 n_users 是为了获取从 0 开始的索引
            clicked_items = [train_user_set[user.item()] - args.n_users for user in batch_users]
            #获取每个用户在测试集中实际交互过的物品
            groundTruth_items = [test_user_set[user.item()] - args.n_users for user in batch_users]
#初始化两个列表用于收集需要在评分矩阵中排除的元素的索引。
            exclude_index = []
            exclude_items = []
            # 对于每个用户，排除他们在训练集中已经交互过的物品，以确保推荐的物品是新的
            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items)) #为每个需要排除的物品添加其用户的索引。
                exclude_items.extend(items) #：添加要排除的物品的索引
#在评分矩阵中，将用户已交互过的物品的评分设置为一个非常低的值（例如负的大数），确保这些物品不会出现在推荐列表的前 k 个位置。
            rating_batch[exclude_index, exclude_items] = -(1 << 10)
            #使用 torch.topk 获取每个用户的前 k 个推荐项
            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))
        #对所有批次的结果进行汇总，计算总的评估指标
        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']
        # 将累加的结果除以测试用户的总数，得到平均的评估指标
        results['Recall'] /= len(test_users)
        results['Precision'] /= len(test_users)
        results['NDCG'] /= len(test_users)
        results['F1'] /= len(test_users)
        results['Hit_ratio'] /= len(test_users)

    return results
