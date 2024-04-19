from torch_geometric.utils import add_remaining_self_loops, degree #给图中的每个节点添加自环，这在某些图神经网络模型中是需要的，以确保每个节点在信息传递过程中也考虑自己的特征
import torch_geometric.transforms as T   # 提供了对图数据进行预处理和转换的工具
import torch
# def index_to_mask(index, size):
#     #创建一个指定大小的全0布尔张量，设备与index相同
#     mask = torch.zeros(size, dtype=torch.bool, device=index.device)
#     mask[index] = 1
#     return mask
