"""数据集下载与处理相关的工具函数。"""
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Compose
from typing import Any, List, Optional, Tuple
import numpy as np
import random
import torch
import os

class BatteryDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 从 .pt 文件中加载特征和标签
        self.features, self.targets = torch.load(data_path)
        self.transform = transform  # 添加 transform 参数以支持数据增强
        self.targets = np.asarray(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = int(self.targets[idx])
        if self.transform:
            feature = self.transform(feature)
        return feature, target

def download_data() -> Tuple[Dataset, Dataset]:
    """下载并返回电池数据集。

    Returns
    -------
    Tuple[Dataset, Dataset]
        用于训练的数据集，以及用于测试的数据集。
    """
    output_dir = './dataset/battery/'
    time_series_transforms = Compose([
        AddNoise(noise_level=0.1),
        RandomShift(shift_max=5),
        RandomScale(scale_min=0.9, scale_max=1.1),
        TimeMask(mask_ratio=0.1),
    ])
    trainset = BatteryDataset(data_path=os.path.join(output_dir, 'train.pt'), transform=time_series_transforms)
    testset = BatteryDataset(data_path=os.path.join(output_dir, 'test.pt'), transform=time_series_transforms)
    return trainset, testset

def subset_data(trainset, testset, subset_list, config) -> Tuple[Dataset, Dataset]:
    if not subset_list:
        return trainset, testset
    class_indices = {label: [] for label in subset_list}
    # 收集每个类别的索引
    for idx, label in enumerate(trainset.targets):
        if label in subset_list:
            class_indices[label].append(idx)
    subset_indices = []
    for indices in class_indices.values():
        subset_indices.extend(indices)
    # 创建子数据集
    train_subset = Subset(trainset, subset_indices)
    train_subset.targets = trainset.targets[subset_indices]
    if config.train.test_all:
        sub = config.dataset_config.subset_list + config.dataset_config.ignore_class
        class_indices = {label: [] for label in sub}
        # 收集每个类别的索引
        for idx, label in enumerate(testset.targets):
            if label in sub:
                class_indices[label].append(idx)
        subset_indices = []
        for indices in class_indices.values():
            subset_indices.extend(indices)
        # 创建子数据集
        test_subset = Subset(testset, subset_indices)
        test_subset.targets = testset.targets[subset_indices]
        return train_subset, test_subset
    else:
        class_indices = {label: [] for label in subset_list}
        # 收集每个类别的索引
        for idx, label in enumerate[Any](testset.targets):
            if label in subset_list:
                class_indices[label].append(idx)
        subset_indices = []
        for indices in class_indices.values():
            subset_indices.extend(indices)
        # 创建子数据集
        test_subset = Subset(testset, subset_indices)
        test_subset.targets = testset.targets[subset_indices]
        return train_subset, test_subset

def partition_data(
    config,
    num_clients,
    iid: Optional[bool] = False,
    subset_list = None,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """将训练集划分为 iid 或 non-iid 分区，以模拟联邦学习场景。
    参数
    ----------
    num_clients : int
        持有部分数据的客户端数量。
    iid : bool, optional
        数据在客户端之间是否独立同分布（iid）。如果为 False，则会先按标签
        对数据进行排序，再按块划分给每个客户端（用于在最坏情况下测试收敛性），默认 False。
    seed : int, optional
        用于复现实验的固定随机种子，默认 42。

    返回
    -------
    Tuple[List[Dataset], Dataset]
        每个客户端对应的训练集列表，以及用于测试模型的单个测试集。
    """
    trainset, testset = download_data()
    trainset, testset = subset_data(trainset, testset, subset_list, config)
    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * (num_clients-1) + [len(trainset) - partition_size * (num_clients-1)]
    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        datasets = dirichlet_split(trainset, num_clients, config.dataset_config.alpha)
    return datasets, testset

def dirichlet_split(
    trainset: Dataset,
    num_partitions: int,
    alpha = 1,
) -> Dataset:
    """按照dirichlet方式对数据集进行划分(参考 Li 等人 2020 的实现)。
    实现参考:Li et al 2020:https://arxiv.org/abs/1812.06127，并使用默认参数设定。

    参数
    ----------
    trainset : Dataset
        训练集。
    num_partitions: int
        划分的分区数量（客户端数）。
    alpha: float
        Dirichlet 分布参数。

    返回
    -------
    Dataset
        分区后的训练集列表。
    """
    targets = trainset.targets
    class_counts = len(np.bincount(trainset.targets))
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    np.random.seed(42)
    label_distribution = np.random.dirichlet([alpha]*num_partitions, class_counts)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(targets == y).flatten() for y in range(class_counts)]
    class_idcs_firstN = [np.argwhere(targets == y).flatten()[:1] for y in range(class_counts)]
    class_idcs_firstN = np.concatenate(class_idcs_firstN, axis=0)
    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(num_partitions)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate([np.concatenate(idcs),class_idcs_firstN]) for idcs in client_idcs]
    # client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    # 构建子集
    partitions = [Subset(trainset, p) for p in client_idcs]
    return partitions

# 增强方法1：添加随机噪声
class AddNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

# 增强方法2：随机时间偏移
class RandomShift:
    def __init__(self, shift_max=5):
        self.shift_max = shift_max

    def __call__(self, x):
        shift = random.randint(-self.shift_max, self.shift_max)
        return torch.roll(x, shifts=shift, dims=0)

# 增强方法3：幅值缩放
class RandomScale:
    def __init__(self, scale_min=0.9, scale_max=1.1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, x):
        scale = random.uniform(self.scale_min, self.scale_max)
        return x * scale

# 增强方法4：时间遮盖
class TimeMask:
    def __init__(self, mask_ratio=0.1):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        mask_num = int(len(x) * self.mask_ratio)
        mask_indices = random.sample(range(len(x)), mask_num)
        x_clone = x.clone()
        x_clone[mask_indices] = 0  # 将遮盖部分设为 0
        return x_clone