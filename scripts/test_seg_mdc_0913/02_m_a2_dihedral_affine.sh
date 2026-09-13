#!/bin/bash
# 条件化 backbone 分割微调（须先完成 run_pretrain_conditioned.sh）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../experiment_naming.sh"

# Experiment T-A2: dihedral plus mild affine augmentation.
EXPERIMENT_ID="M-A2"
AUGMENT="true"
AUGMENTATION_COPIES="1"
AUGMENTATION_POLICY="dihedral_affine"
AUGMENTATION_PROBABILITY="1.0"
AFFINE_ROTATION_DEGREES="15.0"
AFFINE_SCALE_DELTA="0.1"
AFFINE_TRANSLATE_FRACTION="0.05"
SEGMENTATION_HEAD="h0_simple"
SEGMENTATION_LOSS="ce_dice"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# MDC fixed splits (all inputs are 256x256 patches)
TRAIN_ROOT="${TRAIN_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_train_p049_20260728}"
VAL_ROOT="${VAL_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_val_p010_20260728}"
TEST_ROOT="${TEST_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_test_p010_20260728}"

# Conditioned 预训练 checkpoint（手动设置完整路径）
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
EVAL_ONLY_CHECKPOINT="${EVAL_ONLY_CHECKPOINT:-}"

# ── 训练超参数 ────────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-200}"
# 优先使用顺序脚本或命令行导出的 LR；单独执行时保持原默认值。
LR="${LR:-4e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-42}"
AMP="${AMP:-true}"
EARLY_STOP="${EARLY_STOP:-false}"
PATIENCE="${PATIENCE:-20}"

# ── 模型架构（须与预训练保持一致） ───────────────────────────────────────────
NUM_CLASSES="${NUM_CLASSES:-2}"
PATCH_SIZE=16
SPECTRAL_PATCH_SIZE=5
EMBED_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
MLP_RATIO=4.0
DROPOUT=0.1
CNN_STEM_CH=64
CNN_SPECTRAL_AGG="attention"
FUSION_HEADS=8
FEATURE_DIM=128
DECODER_MID_CH=64
RESIDUAL_HIDDEN_DIM=128
RIDGE_LAMBDA=1e-3
CONFIDENCE_TEMPERATURE=0.05
ALPHA_MIN=0.1
ALPHA_EXTRA=1.0
OD_MAX=3.0

# ── NMF 条件输入缓存键（须与预训练保持一致） ─────────────────────────────────
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
ALLOW_INDEX_WAVELENGTHS="${ALLOW_INDEX_WAVELENGTHS:-true}"
WAVELENGTH_FILE="${WAVELENGTH_FILE:-}"          # 真实波长表 .npy；留空时按 ALLOW_INDEX_WAVELENGTHS 回退

# ── 迁移学习 ──────────────────────────────────────────────────────────────────
FREEZE_BACKBONE="${FREEZE_BACKBONE:-false}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-1.0}"
HEAD_LR_MULTIPLIER="${HEAD_LR_MULTIPLIER:-1.0}"

# ── 分割增强 / 分割头 / 损失（默认值严格保持历史基线）───────────────────────
AUGMENT="${AUGMENT:-false}"
AUGMENTATION_COPIES="${AUGMENTATION_COPIES:-1}"
AUGMENTATION_POLICY="${AUGMENTATION_POLICY:-dihedral}"
AUGMENTATION_PROBABILITY="${AUGMENTATION_PROBABILITY:-1.0}"
AFFINE_ROTATION_DEGREES="${AFFINE_ROTATION_DEGREES:-15.0}"
AFFINE_SCALE_DELTA="${AFFINE_SCALE_DELTA:-0.1}"
AFFINE_TRANSLATE_FRACTION="${AFFINE_TRANSLATE_FRACTION:-0.05}"
PERSPECTIVE_SCALE="${PERSPECTIVE_SCALE:-0.05}"
AUGMENTATION_PADDING_MODE="${AUGMENTATION_PADDING_MODE:-reflection}"

SEGMENTATION_HEAD="${SEGMENTATION_HEAD:-h0_simple}"
HEAD_HIDDEN_CHANNELS="${HEAD_HIDDEN_CHANNELS:-128}"
HEAD_PROJECTION_CHANNELS="${HEAD_PROJECTION_CHANNELS:-64}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
ASPP_RATES="${ASPP_RATES:-1,6,12,18}"

SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-ce_dice}"
CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
DICE_LOSS_WEIGHT="${DICE_LOSS_WEIGHT:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.0}"
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.0}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-}"

ENDMEMBER_SCOPE="${ENDMEMBER_SCOPE:-patch}"
SCENE_ENDMEMBER_ROOT="${SCENE_ENDMEMBER_ROOT:-}"

# ── 日志 / 存储 ────────────────────────────────────────────────────────────────
# 默认每个 DDP rank 使用 2 个 worker；可由顺序脚本或命令行安全覆盖。
# 常驻 worker 与预取仅改善供数速度，不改变样本、增强或训练数学语义。
WORKERS="${WORKERS:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DISTRIBUTED_VALIDATION="${DISTRIBUTED_VALIDATION:-true}"
# HD95 计算后端：
#   scipy  -> CPU 距离变换，训练更稳定（推荐默认）
#   monai  -> 与 LoTS-Net reference 一致的 MONAI GPU 实现（本机 DDP 下可能不稳定）
HD95_BACKEND="scipy"
# Dice 指标：默认计算六种附加协议；主指标 batch_allclass_macro 虽未在该
# 列表中显式重复，但 Python 训练入口会自动将其追加，因此验证和测试阶段
# 实际会同时报告全部七种 Dice 协议。
DICE_METRICS="${DICE_METRICS:-fg_binary_scene,micro_fg_scene,weighted_fg_scene,macro_fg_scene,classwise,global_fg}"
PRIMARY_DICE_METRIC="${PRIMARY_DICE_METRIC:-batch_allclass_macro}"
# 测试推理：direct 适合已切好的整齐 patch；sliding_window 用于完整大图。
# TEST_WINDOW_SIZE 是空间推理窗口，不是上面的模型 Token PATCH_SIZE。
TEST_INFERENCE_MODE="${TEST_INFERENCE_MODE:-direct}"
TEST_WINDOW_SIZE="${TEST_WINDOW_SIZE:-256}"
TEST_WINDOW_STRIDE="${TEST_WINDOW_STRIDE:-128}"
TEST_WINDOW_BATCH_SIZE="${TEST_WINDOW_BATCH_SIZE:-4}"
TEST_WINDOW_BLEND="${TEST_WINDOW_BLEND:-gaussian}"
# 仅用于冒烟测试/调试；0 表示使用完整 DataLoader。
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
# 按此 epoch 数划窗口，记录每个窗口内验证集 Dice 最佳的模型，训练结束后
# 逐个在测试集上评估并打印指标，避免验证集全局最优过早出现（loss 还较高时）
# 导致后续 epoch 训练成果被完全忽视。<=0 时禁用窗口机制。
BEST_VAL_INTERVAL="${BEST_VAL_INTERVAL:-10}"
PROGRESS="${PROGRESS:-log}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DATASET_INFO="MDC-256x256-b60-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SAVE_DIR:-./records/test_seg_mdc_0913/${EXPERIMENT_ID}_${DATASET_INFO}_${EXP_TIME}}"
mkdir -p "$SAVE_DIR"
echo "DATASET_INFO=$DATASET_INFO"
echo "SAVE_DIR=$SAVE_DIR"

# ── 可选参数组合 ──────────────────────────────────────────────────────────────
WAVELENGTH_ARG=""
if [ "$ALLOW_INDEX_WAVELENGTHS" = "true" ]; then
    WAVELENGTH_ARG="--allow-index-wavelengths"
fi
WAVELENGTH_FILE_ARG=""
if [ -n "$WAVELENGTH_FILE" ]; then
    WAVELENGTH_FILE_ARG="--wavelength-file $WAVELENGTH_FILE"
fi
SIMPLEX_ARG="--no-nmf-simplex"
if [ "$NMF_SIMPLEX" = "true" ]; then
    SIMPLEX_ARG="--nmf-simplex"
fi
AMP_ARG=""
if [ "$AMP" = "true" ]; then
    AMP_ARG="--amp"
fi
EARLY_STOP_ARG=""
if [ "$EARLY_STOP" = "true" ]; then
    EARLY_STOP_ARG="--early-stop --patience $PATIENCE"
fi
FREEZE_ARG=""
if [ "$FREEZE_BACKBONE" = "true" ]; then
    FREEZE_ARG="--freeze-backbone"
fi
AUGMENT_ARG="--no-augment"
if [ "$AUGMENT" = "true" ]; then
    AUGMENT_ARG="--augment"
fi
SCENE_ENDMEMBER_ARG=""
if [ -n "$SCENE_ENDMEMBER_ROOT" ]; then
    SCENE_ENDMEMBER_ARG="--scene-endmember-root $SCENE_ENDMEMBER_ROOT"
fi
PERSISTENT_WORKERS_ARG=""
if [ "$PERSISTENT_WORKERS" = "true" ]; then
    PERSISTENT_WORKERS_ARG="--persistent-workers"
fi
DISTRIBUTED_VALIDATION_ARG=""
if [ "$DISTRIBUTED_VALIDATION" = "true" ]; then
    DISTRIBUTED_VALIDATION_ARG="--distributed-validation"
fi
EVAL_ONLY_ARGS=()
if [ -n "$EVAL_ONLY_CHECKPOINT" ]; then
    EVAL_ONLY_ARGS+=(--eval-only-checkpoint "$EVAL_ONLY_CHECKPOINT")
fi

MASTER_PORT=$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
")

OMP_NUM_THREADS=2 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_finetune_conditioned.py \
    --train-root             "$TRAIN_ROOT" \
    --val-root               "$VAL_ROOT" \
    --test-root              "$TEST_ROOT" \
    --pretrain-ckpt          "$PRETRAIN_CKPT" \
    --epochs                 $EPOCHS \
    --batch-size             $BATCH_SIZE_PER_GPU \
    --lr                     $LR \
    --min-lr                 $MIN_LR \
    --weight-decay           $WEIGHT_DECAY \
    --warmup-epochs          $WARMUP_EPOCHS \
    --clip-grad              $CLIP_GRAD \
    --seed                   $SEED \
    --num-classes            $NUM_CLASSES \
    --patch-size             $PATCH_SIZE \
    --spectral-patch-size    $SPECTRAL_PATCH_SIZE \
    --nmf-k                  $NMF_K \
    --nmf-l1                 $NMF_L1 \
    --nmf-l2                 $NMF_L2 \
    --nmf-l3                 $NMF_L3 \
    --nmf-lam-e              $NMF_LAM_E \
    --nmf-e-clamp-max        $NMF_E_CLAMP_MAX \
    --embed-dim              $EMBED_DIM \
    --vit-depth              $VIT_DEPTH \
    --vit-heads              $VIT_HEADS \
    --mlp-ratio              $MLP_RATIO \
    --dropout                $DROPOUT \
    --cnn-stem-ch            $CNN_STEM_CH \
    --cnn-spectral-agg       $CNN_SPECTRAL_AGG \
    --fusion-heads           $FUSION_HEADS \
    --feature-dim            $FEATURE_DIM \
    --decoder-mid-ch         $DECODER_MID_CH \
    --residual-hidden-dim    $RESIDUAL_HIDDEN_DIM \
    --ridge-lambda           $RIDGE_LAMBDA \
    --confidence-temperature $CONFIDENCE_TEMPERATURE \
    --alpha-min              $ALPHA_MIN \
    --alpha-extra            $ALPHA_EXTRA \
    --od-max                 $OD_MAX \
    --freeze-backbone-epochs $FREEZE_BACKBONE_EPOCHS \
    --backbone-lr-multiplier $BACKBONE_LR_MULTIPLIER \
    --head-lr-multiplier     $HEAD_LR_MULTIPLIER \
    --segmentation-head      "$SEGMENTATION_HEAD" \
    --head-hidden-channels   $HEAD_HIDDEN_CHANNELS \
    --head-projection-channels $HEAD_PROJECTION_CHANNELS \
    --head-dropout           $HEAD_DROPOUT \
    --aspp-rates             "$ASPP_RATES" \
    --segmentation-loss      "$SEGMENTATION_LOSS" \
    --ce-loss-weight         $CE_LOSS_WEIGHT \
    --dice-loss-weight       $DICE_LOSS_WEIGHT \
    --focal-gamma            $FOCAL_GAMMA \
    --boundary-loss-weight   $BOUNDARY_LOSS_WEIGHT \
    --aux-loss-weight        $AUX_LOSS_WEIGHT \
    --class-weight-mode      "$CLASS_WEIGHT_MODE" \
    --class-weights          "$CLASS_WEIGHTS" \
    --augmentation-copies    $AUGMENTATION_COPIES \
    --augmentation-policy    "$AUGMENTATION_POLICY" \
    --augmentation-probability $AUGMENTATION_PROBABILITY \
    --affine-rotation-degrees $AFFINE_ROTATION_DEGREES \
    --affine-scale-delta     $AFFINE_SCALE_DELTA \
    --affine-translate-fraction $AFFINE_TRANSLATE_FRACTION \
    --perspective-scale      $PERSPECTIVE_SCALE \
    --augmentation-padding-mode "$AUGMENTATION_PADDING_MODE" \
    --endmember-scope        "$ENDMEMBER_SCOPE" \
    --workers                $WORKERS \
    --prefetch-factor         $PREFETCH_FACTOR \
    --save-interval          $SAVE_INTERVAL \
    --best-val-interval      $BEST_VAL_INTERVAL \
    --progress               $PROGRESS \
    --log-interval           $LOG_INTERVAL \
    --hd95-backend           "$HD95_BACKEND" \
    --dice-metrics           "$DICE_METRICS" \
    --primary-dice-metric    "$PRIMARY_DICE_METRIC" \
    --max-train-batches      "$MAX_TRAIN_BATCHES" \
    --max-eval-batches       "$MAX_EVAL_BATCHES" \
    --test-inference-mode    "$TEST_INFERENCE_MODE" \
    --test-window-size       "$TEST_WINDOW_SIZE" \
    --test-window-stride     "$TEST_WINDOW_STRIDE" \
    --test-window-batch-size "$TEST_WINDOW_BATCH_SIZE" \
    --test-window-blend      "$TEST_WINDOW_BLEND" \
    --save-dir               "$SAVE_DIR" \
    $WAVELENGTH_ARG \
    $WAVELENGTH_FILE_ARG \
    $SIMPLEX_ARG \
    $AMP_ARG \
    $EARLY_STOP_ARG \
    $FREEZE_ARG \
    $AUGMENT_ARG \
    $SCENE_ENDMEMBER_ARG \
    $PERSISTENT_WORKERS_ARG \
    $DISTRIBUTED_VALIDATION_ARG \
    "${EVAL_ONLY_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"

