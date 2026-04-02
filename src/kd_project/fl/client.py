"""定义 MNIST 的 Flower 客户端，以及用于实例化客户端的工厂函数。"""
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from flwr.common import Context
from flwr.client.client import Client
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from kd_project.models.models import test, train, load_pretrained_weights
from kd_project.common.utils import calc_weight_acc
import flwr as fl
import numpy as np
import torch

class FlowerClient(
    fl.client.NumPyClient
):
    """CNN 联邦学习的标准 Flower 客户端。"""

    def __init__(
        self,
        config: DictConfig,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        net_bias = None
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule
        self.bias_model = net_bias
        self.config = config

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """返回当前模型参数。"""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """使用给定参数更新模型。"""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_bias_model(self, bias_model_path) -> NDArrays:
        """返回偏差模型参数。"""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """实现客户端的分布式训练函数。"""
        self.set_parameters(parameters)
        # # 定义评估函数，交换模型
        # loss_source, _ = test(self.net, self.valloader , self.device)
        # loss_distillation, _ = test(self.bias_model, self.valloader, self.device)
        # if loss_source > loss_distillation:
        #     net_source = self.bias_model
        #     net_distillation = self.net
        # else:
        net_source = self.net
        net_distillation = self.bias_model
        weight_acc = None
        if self.config.dynamic_weight:
            _ , accuracy_before = test(net_source, self.valloader, self.device)
        train(
            net_source,
            net_distillation,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=float(config["proximal_mu"]),
        )
        if self.config.dynamic_weight:
            _ , accuracy_after = test(net_source, self.valloader, self.device)
            weight_acc = calc_weight_acc(accuracy_before, accuracy_after)
        metrics: Dict[str, Scalar] = {}
        if weight_acc is not None:
            metrics["weight_acc"] = float(weight_acc)
        return self.get_parameters({}), len(self.trainloader), metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """对给定客户端执行分布式评估。"""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    config: DictConfig,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    """生成用于创建 Flower 客户端的客户端工厂函数。

    参数
    ----------
    num_epochs : int
        每个客户端在将更新发送给服务器之前运行的本地训练轮数（epochs）。
    trainloaders: List[DataLoader]
        DataLoader 列表；每个 DataLoader 对应某个客户端的训练数据划分。
    valloaders: List[DataLoader]
        DataLoader 列表；每个 DataLoader 对应某个客户端的验证数据划分。
    learning_rate : float
        客户端 SGD 优化器的学习率。

    返回
    -------
    Callable[[str], FlowerClient]
        用于创建 Flower 客户端的工厂函数。
    """
    def client_fn(context: Context) -> Client:
        """创建表示单个组织的 Flower 客户端。"""
        # 加载模型
        cid = context.node_config.get("partition-id", context.node_id)
        cid_int = int(cid)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        net_bias = None
        if not config.load_params.init:
            load_pretrained_weights(config.save_path_source, net, config)
            if config.load_params.distillation_revise:
                net_bias = instantiate(model).to(device)
                load_pretrained_weights(config.save_path_distillation, net_bias, config)

        # 注意：每个客户端获取不同的训练/验证数据加载器，因此它们各自使用本地数据训练评估。
        trainloader = trainloaders[cid_int]
        valloader = valloaders[cid_int]
        return FlowerClient(
            config,
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            net_bias,
        ).to_client()

    return client_fn


