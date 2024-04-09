自定义 Dataset

尽管 PyG 已经包含许多有用的数据集，我们也可以通过继承 torch_geometric.data.Dataset 使用自己的数据集。提供 2 种不同的 Dataset：

InMemoryDataset：使用这个 Dataset 会一次性把数据全部加载到内存中。
Dataset: 使用这个 Dataset 每次加载一个数据到内存中，比较常用。
我们需要在自定义的 Dataset 的初始化方法中传入数据存放的路径，然后 PyG 会在这个路径下再划分 2 个文件夹：

raw_dir: 存放原始数据的路径，一般是 csv、mat 等格式
processed_dir: 存放处理后的数据，一般是 pt 格式 ( 由我们重写 process() 方法实现)。
在 PyTorch 中，是没有这两个文件夹的。下面来说明一下这两个文件夹在 PyG 中的实际意义和处理逻辑。
torch_geometric.data.Dataset 继承自 torch.utils.data.Dataset，在初始化方法 __init__() 中，会调用_download() 方法和_process() 方法。

```
def __init__(self, root=None, transform=None, pre_transform=None,
    pre_filter=None):
 super(Dataset, self).__init__()

 if isinstance(root, str):
  root = osp.expanduser(osp.normpath(root))

 self.root = root
 self.transform = transform
 self.pre_transform = pre_transform
 self.pre_filter = pre_filter
 self.__indices__ = None

 # 执行 self._download() 方法
 if 'download' in self.__class__.__dict__.keys():
  self._download()
    # 执行 self._process() 方法
 if 'process' in self.__class__.__dict__.keys():
  self._process()
```

_download() 方法如下，首先检查 self.raw_paths 列表中的文件是否存在；如果存在，则返回；如果不存在，则调用 self.download() 方法下载文件。

```
def _download(self):
 if files_exist(self.raw_paths):  # pragma: no cover
  return

 makedirs(self.raw_dir)
 self.download()
```

_process() 方法如下，首先在 self.processed_dir 中有 pre_transform，那么判断这个 pre_transform 和传进来的 pre_transform 是否一致，如果不一致，那么警告提示用户先删除 self.processed_dir 文件夹。pre_filter 同理。

然后检查 self.processed_paths 列表中的文件是否存在；如果存在，则返回；如果不存在，则调用 self.process() 生成文件。

```
def _process(self):
 f = osp.join(self.processed_dir, 'pre_transform.pt')
 if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
  warnings.warn(
   'The `pre_transform` argument differs from the one used in '
   'the pre-processed version of this dataset. If you really '
   'want to make use of another pre-processing technique, make '
   'sure to delete `{}` first.'.format(self.processed_dir))
 f = osp.join(self.processed_dir, 'pre_filter.pt')
 if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
  warnings.warn(
   'The `pre_filter` argument differs from the one used in the '
   'pre-processed version of this dataset. If you really want to '
   'make use of another pre-fitering technique, make sure to '
   'delete `{}` first.'.format(self.processed_dir))

 if files_exist(self.processed_paths):  # pragma: no cover
  return

 print('Processing...')

 makedirs(self.processed_dir)
 self.process()

 path = osp.join(self.processed_dir, 'pre_transform.pt')
 torch.save(__repr__(self.pre_transform), path)
 path = osp.join(self.processed_dir, 'pre_filter.pt')
 torch.save(__repr__(self.pre_filter), path)

 print('Done!')
```

一般来说不用实现 downloand() 方法。

如果你直接把处理好的 pt 文件放在了 self.processed_dir 中，那么也不用实现 process() 方法。

在 Pytorch 的 dataset 中，我们需要实现__getitem__() 方法，根据 index 返回样本和标签。在这里 torch_geometric.data.Dataset 中，重写了__getitem__() 方法，其中调用了 get() 方法获取数据




https://zhuanlan.zhihu.com/p/142948273
