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
    """在 MNIST 上运行 CNN 联邦学习。

    参数
    ----------
    cfg : DictConfig
        存储 Hydra 配置的 omegaconf 对象。
    """
    # 单次 Hydra job 内跑多阶段（不写回 conf/config.yaml）
    base_save_path = HydraConfig.get().runtime.output_dir
    stage1_save_path = os.path.join(base_save_path, "stage1_federated")
    stage2_save_path = os.path.join(base_save_path, "stage2_distillation")
    stage3_save_path = os.path.join(base_save_path, "stage3_federated")
    stage4_save_path = os.path.join(base_save_path, "stage4_federated")
    stage5_save_path = os.path.join(base_save_path, "stage5_distillation")
    stage6_save_path = os.path.join(base_save_path, "stage6_federated")

    # stage1: 联邦训练（准备 teacher）
    cfg1 = copy.deepcopy(cfg)
    cfg1.dataset_config.subset_list = [0,1,2,3,4,6,7]
    cfg1.dataset_config.ignore_class = [5]
    cfg1.model.num_classes = 8
    cfg1.pretrained = False # 不使用预训练模型
    cfg1.train.unique = False # 不使用特殊类型数据处理
    cfg1.load_params.init = True
    cfg1.add_client = False # 不加入通信受限测控中心
    cfg1.load_params.distillation_revise = False # 不使用蒸馏修正
    cfg1.weight = False
    cfg1.dynamic_weight = False
    cfg1.save_path_final= stage1_save_path # 保存路径
    run_federated(cfg1, stage1_save_path)

    # stage2: knowledge distillation（产出偏置模型）
    cfg2 = copy.deepcopy(cfg)
    cfg1.dataset_config.subset_list = [0,1,2,3,4,6,7]
    cfg1.dataset_config.ignore_class = [5]
    cfg2.distillation = True
    cfg2.distillation_adjust = False
    cfg2.train.unique = True
    cfg2.save_path_source = stage1_save_path # 保存路径
    cfg2.save_path_final = stage2_save_path # 保存路径
    run_knowledge_distillation(cfg2, distillation_save_path=stage2_save_path)

    # stage3: 再联邦训练（加载 stage2 的偏置模型）
    cfg3 = copy.deepcopy(cfg)
    cfg3.load_params.init = False
    cfg3.pretrained = True
    cfg3.add_client = True
    cfg3.train.unique = True
    cfg3.load_params.distillation_revise = True
    cfg3.weight = True
    cfg3.dynamic_weight = True
    cfg3.save_path_source = stage1_save_path # 保存路径
    cfg3.save_path_distillation = stage2_save_path # 保存路径
    cfg3.save_path_final = stage3_save_path # 保存路径
    run_federated(cfg3, stage3_save_path)

    # stage4: 再联邦训练（不同 ignore_class / num_classes）
    cfg4 = copy.deepcopy(cfg)
    cfg4.dataset_config.subset_list = [0,1,2,3,4,6,7]
    cfg4.dataset_config.ignore_class = [5,8]
    cfg4.model.num_classes = 9
    cfg4.load_params.full = False
    cfg4.pretrained = True
    cfg4.train.unique = False
    cfg4.load_params.init = False
    cfg4.add_client = False
    cfg4.load_params.distillation_revise = False
    cfg4.weight = False
    cfg4.dynamic_weight = False
    cfg4.save_path_source = stage3_save_path
    cfg4.save_path_distillation = stage2_save_path
    cfg4.save_path_final = stage4_save_path
    run_federated(cfg4, stage4_save_path)

    # stage5: 再 distillation
    cfg5 = copy.deepcopy(cfg)
    cfg5.load_params.full = True
    cfg5.distillation = True
    cfg5.distillation_adjust = False
    cfg5.train.unique = True
    cfg5.save_path_source = stage4_save_path
    cfg5.save_path_final = stage5_save_path
    run_knowledge_distillation(cfg5, distillation_save_path=stage5_save_path)

    # stage6: 最终联邦训练（加载 stage5 的偏置模型）
    cfg6 = copy.deepcopy(cfg)
    cfg6.load_params.init = False
    cfg6.pretrained = True
    cfg6.add_client = True
    cfg6.train.unique = True
    cfg6.load_params.distillation_revise = True
    cfg6.weight = True
    cfg6.dynamic_weight = True
    cfg6.save_path_source = stage4_save_path
    cfg6.save_path_distillation = stage5_save_path
    cfg6.save_path_final = stage6_save_path
    run_federated(cfg6, stage6_save_path)

if __name__ == "__main__":
    main()





