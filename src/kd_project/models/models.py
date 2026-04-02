"""MNIST 的 CNN 模型架构、训练和测试函数。"""
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path
import torch.nn as nn
import torch

class CNNBattery(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 12))
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(3, 12))
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(3, 12))
        self.conv4 = nn.Conv2d(10, 10, kernel_size=(3, 12))
        self.fc1 = nn.Linear(360, 50)
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (1, 3)))
        x = F.relu(F.max_pool2d(self.conv2(x), (1, 3)))
        x = F.relu(F.max_pool2d(self.conv3(x), (1, 3)))
        x = F.relu(F.max_pool2d(self.conv4(x), (1, 3)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(
    net: nn.Module,
    bias_model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:
    """在训练集上训练模型。
    参数
    ----------
    net : nn.Module
        待训练的神经网络。
    bias_model : nn.Module
        用于计算偏差损失的参考模型。
    trainloader : DataLoader
        训练数据加载器。
    device : torch.device
        训练设备，'cpu' 或 'cuda'。
    epochs : int
        训练轮数。
    learning_rate : float
        Adam 优化器学习率。
    proximal_mu : float
        Proximal 项权重系数。
    """
    criterion = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    if bias_model != None:
        bias_model.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, bias_model, global_params, trainloader, device, criterion, mse_loss, optimizer, proximal_mu
        )


def _train_one_epoch(  # pylint: disable=too-many-arguments
    net: nn.Module,
    bias_model,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    mse_loss,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:
    """训练一个 epoch。

    参数
    ----------
    net : nn.Module
        待训练的网络。
    bias_model
        偏差模型（用于额外 MSE 约束），可为 None。
    global_params : List[Parameter]
        来自服务器的全局模型参数。
    trainloader : DataLoader
        训练数据加载器。
    device : torch.device
        训练设备。
    criterion : torch.nn.CrossEntropyLoss
        交叉熵损失函数。
    mse_loss
        均方误差损失函数。
    optimizer : torch.optim.Adam
        优化器。
    proximal_mu : float
        Proximal 项权重。

    返回
    -------
    nn.Module
        更新后的模型。
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        if bias_model != None:
            with torch.no_grad():
                bias_pre = bias_model(images)
        optimizer.zero_grad()
        pre = net(images)
        if bias_model != None:
            bias_loss = mse_loss(bias_pre, pre)
        else:
            bias_loss = 0
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = 0.55*criterion(pre, labels) + (proximal_mu / 2) * proximal_term + 0.45*bias_loss
        loss.backward()
        optimizer.step()
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """在完整测试集上评估模型。

    参数
    ----------
    net : nn.Module
        待测试网络。
    testloader : DataLoader
        测试数据加载器。
    device : torch.device
        测试设备。

    返回
    -------
    Tuple[float, float]
        模型在测试集上的 loss 和 accuracy。
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def test_more(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """在完整测试集上评估模型并返回标签/预测信息。

    参数
    ----------
    net : nn.Module
        待测试网络。
    testloader : DataLoader
        测试数据加载器。
    device : torch.device
        测试设备。

    返回
    -------
    Tuple[float, float]
        模型 loss、accuracy、真实标签和预测结果。
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    pred = []
    lab = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            pred.extend(list(predicted.cpu()))
            lab.extend(list(labels.cpu()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy, lab, pred

def load_pretrained_weights(path,net,config):
    """加载预训练权重。"""
    # 根据配置决定加载方式，并加载模型参数
    state_dict = torch.load(Path(path) / Path('model.pkl'))
    if config.load_params.full:
        # 如果参数形状匹配
        net.load_state_dict(state_dict, strict=True)
    else:
        if config.load_params.freeze:
            # 冻结前几类参数
            net.load_state_dict(state_dict, strict=False)
            for name, param in net.named_parameters():
                param.requires_grad = False
            net.linear_head2.weight.requires_grad = True
            net.linear_head2.bias.requires_grad = True

        else:
            for key in list(state_dict.keys()):
                if 'linear_head1' or 'linear_head2' in key:  # 假设输出层是 classifier 的第 3 层
                    del state_dict[key]
            net.load_state_dict(state_dict, strict=False)
    print('load params')