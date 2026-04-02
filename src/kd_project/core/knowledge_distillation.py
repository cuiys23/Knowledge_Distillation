from sklearn.metrics import confusion_matrix
from kd_project.models.models import load_pretrained_weights
from kd_project.models.simple_vit import ModifiedSimpleViT
from hydra.utils import instantiate
from kd_project.data.dataset import load_datasets
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from kd_project.common.utils import caculate_acc
import torch.optim as optim
from pathlib import Path
import seaborn as sns
import torch.nn as nn
import logging
import pickle
import torch
import os

torch.manual_seed(42)

def _build_modified_vit(cfg: DictConfig, device: torch.device) -> ModifiedSimpleViT:
    return ModifiedSimpleViT(
        num_classes=cfg.model.num_classes,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim,
        channels=cfg.model.channels,
    ).to(device)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_result_and_model(save_dir: Path, data: dict, model: torch.nn.Module) -> None:
    _ensure_dir(save_dir)
    with open((save_dir / Path("result.pkl")), "wb") as f:
        pickle.dump(data, f)
    torch.save(model.state_dict(), (save_dir / Path("model.pkl")))


def _plot_accuracy_and_confusion_matrix(
    accuracy_list: list,
    cm,
    lab: list,
    pred: list,
    ignore_class,
) -> None:
    plt.plot(range(len(accuracy_list)), accuracy_list, linestyle="-")
    plt.show()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    acc, acc_ignore = caculate_acc(lab, pred, ignore_class)
    plt.text(1, 1, f"acc: {acc}")
    plt.text(5, 5, f"acc_ignore: {acc_ignore}")
    plt.show()


def _forward_logits(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def evaluate(model: torch.nn.Module, data_loader, device: torch.device):
    """评估分类模型并返回 (loss, accuracy, lab, pred)。

    兼容两种 forward 形式：
    - model(x) -> logits
    - model(x) -> (logits, feature_map)
    """
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    pred = []
    lab = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = _forward_logits(model, inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits.data, 1)
            pred.extend(list(predicted.cpu()))
            lab.extend(list(labels.cpu()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(data_loader.dataset) == 0:
        raise ValueError("Dataloader dataset can't be 0, exiting...")
    loss /= len(data_loader.dataset)
    accuracy = correct / total
    return loss, accuracy, lab, pred


def run_knowledge_distillation(cfg: DictConfig, distillation_save_path: str) -> None:
    os.makedirs(distillation_save_path, exist_ok=True)
    ignore_class = cfg.dataset_config.ignore_class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 划分数据集并获取 DataLoader
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg,
        num_clients=cfg.num_clients+1,
        batch_size=cfg.batch_size,
    )
    train_loader, val_loader = trainloaders[cfg.train.unique_one], valloaders[cfg.train.unique_one]
    path = Path(cfg.save_path_source)
    save_path_distillation = Path(distillation_save_path)
    # 普通监督训练
    if cfg.distillation == "base_train":
        net = instantiate(cfg.model).to(device)
        # 加载预训练权重
        test_loss_list, train_loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list = train(
            net,
            train_loader,
            testloader,
            cfg.num_rounds,
            cfg.learning_rate,
            device,
            ignore_class=ignore_class,
        )
        _, _, lab, pred = evaluate(net, testloader, device)
        cm = confusion_matrix(lab, pred)
        data = {
            'test_loss': test_loss_list,
            'accuracy': accuracy_list,
            'train_loss': train_loss_list,
            'confusion_matrix': cm,
            'accuracy_unignore': accuracy_unignore_list,
            'accuracy_ignore': accuracy_ignore_list,
        }
        _save_result_and_model(save_path_distillation, data, net)
        _plot_accuracy_and_confusion_matrix(accuracy_list, cm, lab, pred, ignore_class)
    # “带基线约束”的自适应权重特征蒸馏
    elif cfg.distillation == "distillation_adjust":
        net = instantiate(cfg.model).to(device)
        load_pretrained_weights(path, net, cfg)
        test_loss_list, train_loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list = train(net,
                                                                                                             train_loader,
                                                                                                             testloader,
                                                                                                             cfg.num_rounds,
                                                                                                             cfg.learning_rate,
                                                                                                             device,
                                                                                                             ignore_class=ignore_class)
        modified_nn_teacher_reg = _build_modified_vit(cfg, device)
        load_pretrained_weights(path, modified_nn_teacher_reg, cfg)
        modified_nn_student_reg = _build_modified_vit(cfg, device)
        load_pretrained_weights(path, modified_nn_student_reg, cfg)
        loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list = train_mse_loss_adjust(
            teacher=modified_nn_teacher_reg,
            student=modified_nn_student_reg,
            base=net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=testloader,
            epochs=cfg.num_rounds,
            learning_rate=cfg.learning_rate,
            feature_map_weight=0.5,
            ce_loss_weight=0.5,
            device=device,
            ignore_class=ignore_class,
        )
        lab,pred,_,_ = test_multiple_outputs(modified_nn_student_reg, testloader, device)
        cm = confusion_matrix(lab, pred)
        data = {'loss': loss_list, 'accuracy': accuracy_list, 'confusion_matrix': cm, 'accuracy_unignore': accuracy_unignore_list, 'accuracy_ignore': accuracy_ignore_list}
        _save_result_and_model(save_path_distillation, data, modified_nn_student_reg)
        _plot_accuracy_and_confusion_matrix(accuracy_list, cm, lab, pred, ignore_class)
    elif cfg.distillation == "distillation":
        # 固定权重的特征蒸馏
        modified_nn_teacher_reg = _build_modified_vit(cfg, device)
        load_pretrained_weights(path, modified_nn_teacher_reg, cfg)
        modified_nn_student_reg = _build_modified_vit(cfg, device)
        load_pretrained_weights(path, modified_nn_student_reg, cfg)
        loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list = train_mse_loss(
            teacher=modified_nn_teacher_reg,
            student=modified_nn_student_reg,
            train_loader=train_loader,
            test_loader=testloader,
            epochs=cfg.num_rounds,
            learning_rate=cfg.learning_rate,
            feature_map_weight=0.3,
            ce_loss_weight=0.7,
            device=device,
            ignore_class=ignore_class,
        )
        lab,pred,_,_ = test_multiple_outputs(modified_nn_student_reg, testloader, device)
        cm = confusion_matrix(lab, pred)
        data = {'loss': loss_list, 'accuracy': accuracy_list, 'confusion_matrix': cm, 'accuracy_unignore': accuracy_unignore_list, 'accuracy_ignore': accuracy_ignore_list}
        _save_result_and_model(save_path_distillation, data, modified_nn_student_reg)
        _plot_accuracy_and_confusion_matrix(accuracy_list, cm, lab, pred, ignore_class)
    else:
        print("没有该训练方式")

def train(net, train_loader, testloader, epochs, learning_rate, device, ignore_class):
    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)
    net.to(device)
    net.train()
    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    accuracy_unignore_list = []
    accuracy_ignore_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pre = net(inputs)
            # 计算真实标签损失
            loss = CEloss(pre, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        train_loss_list.append(running_loss)
        loss, accuracy, lab, pred = evaluate(net, testloader, device)
        accuracy_unignore, accuracy_ignore = caculate_acc(lab, pred, ignore_class)
        test_loss_list.append(loss)
        accuracy_list.append(accuracy)
        accuracy_unignore_list.append(accuracy_unignore)
        accuracy_ignore_list.append(accuracy_ignore)
        logging.info(f"Epoch {epoch + 1}/{epochs}, accuracy: {accuracy}")
    return test_loss_list, train_loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list


def test_multiple_outputs(model, test_loader, device):
    loss, accuracy, lab, pred = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    return lab,pred,loss,accuracy
def train_mse_loss(teacher, student, train_loader, test_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight,
                   device, ignore_class):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师模型设为评估模式
    student.train()  # 学生模型设为训练模式
    loss_list = []
    accuracy_list = []
    accuracy_unignore_list = []
    accuracy_ignore_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 再次忽略教师模型输出的 logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # 学生模型前向传播
            student_logits, regressor_feature_map = student(inputs)
            # 计算特征图损失
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
            # 计算真实标签损失
            label_loss = ce_loss(student_logits, labels)
            # 两个损失加权求和
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        lab, pred, loss, accuracy = test_multiple_outputs(student, test_loader, device)
        accuracy_unignore, accuracy_ignore = caculate_acc(lab, pred, ignore_class)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        accuracy_unignore_list.append(accuracy_unignore)
        accuracy_ignore_list.append(accuracy_ignore)
    return loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list

def train_mse_loss_adjust(teacher, student, base, train_loader, val_loader, test_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight,
                   device, ignore_class):
    base.to(device)
    base.eval()
    ce_loss = nn.CrossEntropyLoss()
    _, _, lab, pred = evaluate(base, val_loader, device)
    base_unignore, base_ignore = caculate_acc(lab, pred, ignore_class)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师模型设为评估模式
    student.train()  # 学生模型设为训练模式
    loss_list = []
    accuracy_list = []
    accuracy_unignore_list = []
    accuracy_ignore_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 再次忽略教师模型输出的 logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)
            # 使用学生模型进行前向传播
            student_logits, regressor_feature_map = student(inputs)
            # 计算损失
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
            # 计算真实标签对应的损失
            label_loss = ce_loss(student_logits, labels)
            # 对两项损失进行加权求和
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        lab, pred, loss, accuracy = test_multiple_outputs(student, val_loader, device)
        accuracy_unignore, accuracy_ignore = caculate_acc(lab, pred, ignore_class)
        if accuracy_unignore >= base_unignore:
            delta = 0.0001
        else:
            delta = -0.0001
        print(f'base_unignore:{base_unignore},base_ignore:{base_ignore},accuracy_ignore:{accuracy_ignore},accuracy_unignore:{accuracy_unignore}')
        feature_map_weight, ce_loss_weight = feature_map_weight+delta, ce_loss_weight-delta
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        lab, pred, loss, accuracy = test_multiple_outputs(student, test_loader, device)
        accuracy_unignore, accuracy_ignore = caculate_acc(lab, pred, ignore_class)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        accuracy_unignore_list.append(accuracy_unignore)
        accuracy_ignore_list.append(accuracy_ignore)
    return loss_list, accuracy_list, accuracy_unignore_list, accuracy_ignore_list

if __name__ == "__main__":
    pass