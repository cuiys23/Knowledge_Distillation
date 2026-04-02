from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import flwr as fl
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix

from kd_project.fl import client, server
from kd_project.common import utils
from kd_project.data.dataset import load_datasets
from kd_project.models.models import test_more


def setup_cuda_env() -> None:
    """统一设置 CUDA 可见设备，避免多入口脚本重复相同样板代码。"""
    # 指定 CUDA 设备的编号顺序按主板的 PCI 总线 ID
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 指定 CUDA 设备可见的编号
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def run_federated(
    cfg: DictConfig,
    save_path: str,
) -> None:
    """联邦训练主流程（shared）。"""
    os.makedirs(save_path, exist_ok=True)

    # 划分数据集并获取 DataLoader
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg,
        num_clients=cfg.num_clients + 1,
        batch_size=cfg.batch_size,
    )

    # 是否增加通信受限测控中心
    if cfg.add_client:
        cfg.num_clients = cfg.num_clients + 1
    else:
        trainloaders = trainloaders[1:]
        valloaders = valloaders[1:]

    # 准备用于生成每个客户端的函数
    client_fn = client.gen_client_fn(
        config=cfg,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    )

    # 获取由策略 evaluate() 调用的评估函数
    device = cfg.server_device
    evaluate_fn = server.gen_evaluate_fn(testloader, device=device, cfg=cfg)

    # 获取用于构建客户端 fit() 接收配置的函数
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # cfg.fit_config 在 conf/config.yaml 中可配置（已新增默认值）
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config["curr_round"] = server_round  # 添加轮次信息
            return fit_config

        return fit_config_fn

    # 根据配置实例化策略。这里传入仅在运行时定义的参数。
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
    )

    # 开始仿真
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    # 实验完成。保存结果并绘制图表。
    print("................")

    # 将结果保存为 pickle，目录由 save_path 指定
    net = instantiate(cfg.model)
    state_dict = torch.load(Path(save_path) / Path("model.pkl"))
    net.load_state_dict(state_dict, strict=True)

    _, _, lab, pred = test_more(net, testloader, device)
    acc, acc_ignore = utils.caculate_acc(lab, pred, cfg.dataset_config.ignore_class)
    cm = confusion_matrix(lab, pred)
    utils.save_results_as_pickle(
        history,
        file_path=save_path,
        extra_results={"confusion_matrix": cm},
    )

    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_iid' if cfg.dataset_config.iid else ''}"
        f"{'_alpha' if cfg.dataset_config.alpha else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        file_suffix,
    )

    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.text(1, 1, f"acc: {acc}")
    plt.text(5, 5, f"acc_ignore: {acc_ignore}")
    plt.show()


if __name__ == "__main__":
    # 这个模块通常由 main 入口脚本调用
    raise SystemExit("Use from main.py / main_streaming.py")

