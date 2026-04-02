"""Flower 服务端。"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from models import test_more
from omegaconf import DictConfig
from pathlib import Path
from utils import caculate_acc

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """生成集中式评估函数。

    参数
    ----------
    testloader : DataLoader
        用于测试模型的数据加载器。
    device : torch.device
        用于评估的设备。

    返回
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        集中式评估函数。
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """使用整个测试集进行评估。"""
        net = instantiate(cfg.model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)
        if server_round == cfg.num_rounds:
            torch.save(net.state_dict(), Path(cfg.save_path_final)/ Path('model.pkl'))
        loss, accuracy, lab, pred = test_more(net, testloader, device=device)
        accuracy_unignore,accuracy_ignore = caculate_acc(lab, pred, cfg.dataset_config.ignore_class)
        return loss, {"accuracy": accuracy, "accuracy_unignore": accuracy_unignore, "accuracy_ignore": accuracy_ignore}

    return evaluate
