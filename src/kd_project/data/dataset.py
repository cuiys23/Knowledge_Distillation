"""联邦学习的 MNIST 数据集工具函数。"""
from kd_project.data.dataset_preparation import partition_data, download_data, subset_data
from torch.utils.data import DataLoader, random_split, ConcatDataset
from typing import Optional, Any
from omegaconf import DictConfig
import torch

def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.2,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> tuple[list[DataLoader[Any]], list[DataLoader[Any]], DataLoader[Any]]:
    """创建用于模型训练、验证和测试的 DataLoader。
    参数
    ----------
    config: DictConfig
        控制数据划分过程的配置。
    num_clients : int
        持有数据的客户端数量。
    val_ratio : float, optional
        用于验证的数据占训练数据的比例(0~1)，默认为 0.2。
    batch_size : Optional[int], optional
        批大小，默认为 32。
    seed : Optional[int], optional
        随机种子，用于复现实验，默认为 42。

    返回
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        训练、验证、测试的 DataLoader。
    """
    datasets, testset = partition_data(
        config,
        num_clients,
        iid=config.dataset_config.iid,
        subset_list = config.dataset_config.subset_list,
        seed=seed,
    )
    if config.train.unique:
        trainset1, testset1 = download_data()
        ignore_class = config.dataset_config.ignore_class
        trainset_add, _ = subset_data(trainset1, testset1, ignore_class, config)
        # 为特殊客户端增加类型数据
        datasets[config.train.unique_one] = ConcatDataset([datasets[config.train.unique_one], trainset_add])
    # 将每个客户端数据划分为训练/验证集并创建 DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)

