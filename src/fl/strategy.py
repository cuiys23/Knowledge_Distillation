"""Flower 联邦学习策略。"""
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from flwr.common.logger import log
from flwr.common import Metrics
from omegaconf import DictConfig, OmegaConf
from functools import reduce
from logging import WARNING
from pathlib import Path
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    Scalar,
    parameters_to_ndarrays,
    Parameters,
    NDArrays)
import numpy as np
import torch

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """在评估阶段使用权重平均聚合指标。
    参数
    ----------
    metrics : List[Tuple[int, Metrics]]
        需要聚合的指标列表。
    返回
    -------
    Metrics
        聚合后的加权平均指标。
    """
    # 将每个客户端的准确率乘以其样本数
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # 聚合并返回自定义指标（加权平均）
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

def aggregate(
    weights_results: List[Tuple[NDArrays, float]],
    acc_weights_results: Optional[List[Tuple[NDArrays, float]]],
) -> NDArrays:
    """计算参数的加权平均值。"""
    if acc_weights_results == None:
        results = weights_results
    else:
        results = [(params, w1*w2) for (params, w1), (_, w2) in zip(weights_results, acc_weights_results)]
    num_examples_total = sum([num_examples for _, num_examples in results])

    # 创建一个权重列表，每个权重都乘以对应的样本数
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # 计算每层的平均权重
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

class FedAvgWithKnowledge(FedProx):
    """自定义 FedProx:在聚合时可处理 straggler 的策略。"""

    def __init__(self, cfg: Optional[DictConfig] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # 优先使用运行时传入配置，避免固定读取 conf/config.yaml 导致阶段配置失效
        self.cfg = cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}
        # 若存在失败且不接受失败，则不聚合
        if not self.accept_failures and failures:
            return None, {}

        # 使用 enumerate() 获取 client id 为 '0' 的索引
        config = self.cfg
        index = next((i for i, (client_proxy, _) in enumerate(results) if client_proxy.cid == '0'), -1)
        weights_results: List[Tuple[NDArrays, float]] = []
        acc_weights_results: Optional[List[Tuple[NDArrays, float]]] = None
        if config.weight:
            if index != -1:
                state_dict = torch.load(Path(config.save_path_distillation) / Path('model.pkl'))
                parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in state_dict.items()])
                # 转换结果
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), float(fit_res.num_examples))
                    for client_proxy, fit_res in results if client_proxy.cid != '0'
                ]
                mean_samples = np.mean([samples for parameters, samples in weights_results])
                weights_results.append((parameters_to_ndarrays(parameters), float(mean_samples)))
                if config.dynamic_weight:
                    acc_weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), float(fit_res.metrics.get("weight_acc", 1.0)))
                    for client_proxy, fit_res in results if client_proxy.cid != '0']
                    mean_acc_weights = float(np.mean([acc_ratio for _, acc_ratio in acc_weights_results])) * (float(config.num_rounds) * 1.8 - server_round) / (float(config.num_rounds) * 1.8)
                    acc_weights_results.append((parameters_to_ndarrays(parameters), float(mean_acc_weights)))
            else:
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), float(fit_res.num_examples))
                    for _, fit_res in results
                ]
                if config.dynamic_weight:
                    acc_weights_results = [
                        (parameters_to_ndarrays(fit_res.parameters), float(fit_res.metrics.get("weight_acc", 1.0)))
                        for _, fit_res in results
                    ]
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), float(fit_res.num_examples))
                for _, fit_res in results
            ]
            if config.dynamic_weight:
                acc_weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), float(fit_res.metrics.get("weight_acc", 1.0)))
                    for _, fit_res in results
                ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results,acc_weights_results))

        # 如果提供了指标聚合函数，则聚合自定义指标
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # 仅记录一次警告
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

