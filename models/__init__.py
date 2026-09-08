from models.finetune_model_vit import FinetuneModel, SegmentationHead
from models.nmf_pretrain_model_vit import NMFPretrainModel, seq_to_grid
from models.finetune_model_cinet import FinetuneModelCINET
from models.nmf_pretrain_model_cinet import NMFPretrainModelCINET
from models.conditioned_contracts import ConditionedModelConfig
from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from models.finetune_model_conditioned import ConditionedFinetuneModel
from models.finetune_model_conditioned_detection import ConditionedDetectionModel

__all__ = [
    # ViT-only 版本
    "NMFPretrainModel",
    "FinetuneModel",
    "SegmentationHead",
    "seq_to_grid",
    # CINET 版本（CNN ContextualEncoder + ViT + CIAM 双路交叉注意力）
    "NMFPretrainModelCINET",
    "FinetuneModelCINET",
    # 逐图端元条件化 + 残差丰度路径
    "ConditionedModelConfig",
    "EndmemberConditionedPretrainModel",
    "ConditionedFinetuneModel",
    "ConditionedDetectionModel",
]
