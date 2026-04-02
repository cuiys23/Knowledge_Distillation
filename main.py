"""运行电池数据集的联邦学习。"""
import copy
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kd_project.core.knowledge_distillation import run_knowledge_distillation
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import hydra

from kd_project.core.federated_runner import run_federated, setup_cuda_env

setup_cuda_env()

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """单次 Hydra job 内顺序运行:stage1 联邦训练 -> stage2 蒸馏 -> stage3 再训练。"""
    base_save_path = HydraConfig.get().runtime.output_dir
    stage1_save_path = os.path.join(base_save_path, "stage1_federated")
    stage2_save_path = os.path.join(base_save_path, "stage2_distillation")
    stage3_save_path = os.path.join(base_save_path, "stage3_federated")

    # stage1: 基础联邦训练（产出 teacher）
    cfg1 = copy.deepcopy(cfg)
    cfg1.train.unique = False # 不使用特殊类型数据处理
    cfg1.add_client = False # 通信受限测控中心此时不加入
    cfg1.load_params.init = True # 初始化模型参数
    cfg1.save_path_final= stage1_save_path # 保存路径
    run_federated(cfg1, stage1_save_path)

    # stage2: 知识蒸馏（产出 bias/学生模型）
    cfg2 = copy.deepcopy(cfg)
    cfg2.distillation = "distillation" # 进行知识蒸馏
    cfg2.train.unique = True # 使用特殊类型数据处理
    cfg2.learning_rate = 0.0001 # 调低学习率
    cfg2.save_path_source = stage1_save_path # 保存路径
    cfg2.save_path_final = stage2_save_path # 保存路径
    run_knowledge_distillation(cfg2, distillation_save_path=stage2_save_path)

    # stage3: 使用 stage1 teacher + stage2 distillation_revise 进行再训练（产出 final）
    cfg3 = copy.deepcopy(cfg)
    cfg3.train.unique = False # 不使用特殊类型数据处理
    cfg3.add_client = True # 通信受限测控中心(虚拟客户端)加入
    cfg3.load_params.init = False # 不初始化模型参数
    cfg3.load_params.distillation_revise = True # 使用蒸馏修正
    cfg3.weight = True # 使用权重
    cfg3.dynamic_weight = True # 使用动态权重
    cfg3.learning_rate = 0.0001 # 调低学习率
    cfg3.save_path_source = stage1_save_path # 保存路径
    cfg3.save_path_distillation = stage2_save_path # 保存路径
    cfg3.save_path_final = stage3_save_path # 保存路径
    run_federated(cfg3, stage3_save_path)

if __name__ == "__main__":
    main()
