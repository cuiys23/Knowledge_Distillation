"""用于 MNIST 上 CNN 联邦学习的工具函数集合。"""
from flwr.server.history import History
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
from secrets import token_hex
from pathlib import Path
import numpy as np
import pickle

def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    suffix: Optional[str] = "",
) -> None:
    """从 Flower 服务历史记录中绘制指标。
    参数
    ----------
    hist : History
        包含所有轮次评估结果的对象。
    save_plot_path : str
        保存图表的文件夹路径。
    suffix: Optional[str]
        可选字符串，附加到输出文件名末尾。
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    _, values = zip(*metric_dict["accuracy"])

    # 绘制损失
    rounds_loss, values_loss = zip(*hist.losses_centralized)

    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    plt.xlabel("Rounds")
    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.show()


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = None,
    default_filename: str = "results2_5.pkl",
) -> None:
    """将模拟结果保存为 pickle 文件。

    参数
    ----------
    history: History
        由 start_simulation 返回的历史记录对象。
    file_path: Union[str, Path]
        用于创建并存储 history 和 extra_results 的路径。
        如果路径是目录，则使用 default_filename。
        如果路径不存在，会自动创建。如果文件已存在，会在文件名
        后添加一个随机后缀，避免覆盖已有结果。
    extra_results : Optional[Dict]
        要额外保存到磁盘的结果字典。默认：{}（空字典）。
    default_filename: Optional[str]
        当 file_path 指向目录而不是文件时使用的默认文件名。
        默认："results2_5.pkl"。
    """
    path = Path(file_path)
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """为文件名添加随机后缀（避免覆盖原文件）。
            """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """将默认文件名附加到路径末尾。"""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
            # 文件已存在
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history}
    if extra_results is not None:
        data = {**data, **extra_results}

    # 将结果保存为 pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def caculate_acc(lab_list,pred_list,ignore_class):
    """计算两个分类准确率：忽略类外的准确率和忽略类内的准确率。

    参数
    ----------
    lab_list: list
        真实标签序列。
    pred_list: list
        预测标签序列。
    ignore_class: list or set
        需要单独统计的类别集合。

    返回
    ----------
    tuple
        (acc, acc_ignore)，分别为排除 ignore_class 的准确率和仅包含 ignore_class 的准确率。
    """
    y_true = [true for true in lab_list if true not in ignore_class]
    y_pred = [pred for true, pred in zip(lab_list, pred_list) if true not in ignore_class]
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    acc = correct / len(y_true)
    # 筛选出不等于 ignore_class 的元素
    filtered_y_true = [true for true in lab_list if true in ignore_class]
    if len(filtered_y_true) == 0:      # 如果没有需要单独统计的类别，直接返回 acc 和 1.0（表示 acc_ignore 不适用）
        return acc, 1.0
    filtered_y_pred = [pred for true, pred in zip(lab_list, pred_list) if true in ignore_class]
    correct = sum(1 for true, pred in zip(filtered_y_true, filtered_y_pred) if true == pred)
    acc_ignore = correct / len(filtered_y_true)
    return acc,acc_ignore

# 用于计算正确率权重
def calc_weight_acc(accuracy_before, accuracy_after):
    """计算两个准确率数值的加权（简单求和）。

    参数
    ----------
    accuracy_before: float
        先前准确率。
    accuracy_after: float
        后续准确率。

    返回
    ----------
    float
        结果权重准确率（简单相加）。
    """
    accuracy_before = accuracy_before
    accuracy_after = accuracy_after
    weight_acc = accuracy_before + accuracy_after
    return weight_acc