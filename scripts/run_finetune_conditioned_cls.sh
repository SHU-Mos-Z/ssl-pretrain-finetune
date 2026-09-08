#!/bin/bash
# Conditioned backbone patch-level classification fine-tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
source "scripts/experiment_naming.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# Prepare these roots with scripts/run_split_pretrain_finetune.sh using
# KIND=classification. Update the dated paths after materialization.

# 2018WBC（分类）
# TRAIN_ROOT="data/2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50_finetune_train_p049_20260728"
# VAL_ROOT="data/2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50_finetune_val_p011_20260728"
# TEST_ROOT="data/2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50_finetune_test_p011_20260728"

# PLGC（分类）
TRAIN_ROOT="${TRAIN_ROOT:-data/PLGC_class_patch_512x512_overlap_0x0_to_256x256_minmax_finetune_train_p070_20260728}"
VAL_ROOT="${VAL_ROOT:-data/PLGC_class_patch_512x512_overlap_0x0_to_256x256_minmax_finetune_val_p015_20260728}"
TEST_ROOT="${TEST_ROOT:-data/PLGC_class_patch_512x512_overlap_0x0_to_256x256_minmax_finetune_test_p015_20260728}"

# Leave empty for the fully supervised scratch baseline.
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
CLASS_MAP_FILE=""

# The current data root contains B/E/L/M/N (five classes). Change this only
# after the planned sixth class has been added to all splits.
# NUM_CLASSES=5
NUM_CLASSES=3

# Training hyperparameters.
EPOCHS=100
LR="${LR:-4e-4}"
BACKBONE_LR_MULT=0.1
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
CLIP_GRAD=1.0
LABEL_SMOOTHING=0.0
CLASS_WEIGHTING="none"
SAMPLING_STRATEGY="standard"
# H0: h0_gap_linear          — Z + GAP + Linear（原始基线）
# H1: h1_attention_mlp       — Z + attention pooling + MLP
# H2: h2_dual_scale          — f_low + Z 双尺度融合
# H3: h3_multiscale_gated    — f_low + D2 + Z 注意力池化与尺度门控
CLASSIFICATION_HEAD="h0_gap_linear"
HEAD_PROJECTION_DIM=64
HEAD_HIDDEN_DIM=128
HEAD_DROPOUT=0.1
SEED=42
AMP=true
AUGMENT=true
# 每个原始训练样本在每个 epoch 中产生的在线虚拟视图数（1~8）。
# combined 策略下，同一原样本的多个视图使用互不重复的旋转/翻转，
# 并以给定概率叠加随机四点透视。CLI 默认仍为 dihedral，以兼容旧实验。
AUGMENTATION_COPIES=1
AUGMENTATION_POLICY="dihedral_perspective"
PERSPECTIVE_PROBABILITY=0.5
PERSPECTIVE_SCALE=0.05
PERSPECTIVE_PADDING_MODE="reflection"
EARLY_STOP=false
PATIENCE=20
FREEZE_BACKBONE=false

# Backbone architecture: must match conditioned pre-training.
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

# NMF cache key: must match offline NMF and pre-training.
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
ALLOW_INDEX_WAVELENGTHS=true
WAVELENGTH_FILE=""

WORKERS=1
SAVE_INTERVAL=10
# 按此 epoch 数划窗口，记录每个窗口内验证集 Macro-F1 最佳的模型，训练结束后
# 逐个在测试集上评估并打印指标，避免验证集全局最优过早出现（loss 还较高时）
# 导致后续 epoch 训练成果被完全忽视。<=0 时禁用窗口机制。
BEST_VAL_INTERVAL=10
PROGRESS="log"
LOG_INTERVAL=10
DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}-${CLASSIFICATION_HEAD}-aug${AUGMENTATION_COPIES}-${AUGMENTATION_POLICY}-pp${PERSPECTIVE_PROBABILITY}-ps${PERSPECTIVE_SCALE}"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/finetune_conditioned_cls/${DATASET_INFO}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
echo "DATASET_INFO=$DATASET_INFO"
echo "SAVE_DIR=$SAVE_DIR"
echo "AUGMENTATION=$AUGMENT copies=$AUGMENTATION_COPIES policy=$AUGMENTATION_POLICY"
echo "PERSPECTIVE probability=$PERSPECTIVE_PROBABILITY scale=$PERSPECTIVE_SCALE padding=$PERSPECTIVE_PADDING_MODE"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ ! -d "$root" ]; then
        echo "Missing classification split: $root"
        echo "Run scripts/run_split_pretrain_finetune.sh with KIND=classification first."
        exit 1
    fi
done
if [ -n "$PRETRAIN_CKPT" ] && [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing conditioned pre-training checkpoint: $PRETRAIN_CKPT"
    exit 1
fi

OPTIONAL_ARGS=()
if [ -n "$PRETRAIN_CKPT" ]; then OPTIONAL_ARGS+=(--pretrain-ckpt "$PRETRAIN_CKPT"); fi
if [ -n "$CLASS_MAP_FILE" ]; then OPTIONAL_ARGS+=(--class-map-file "$CLASS_MAP_FILE"); fi
if [ -n "$WAVELENGTH_FILE" ]; then OPTIONAL_ARGS+=(--wavelength-file "$WAVELENGTH_FILE"); fi
if [ "$ALLOW_INDEX_WAVELENGTHS" = "true" ]; then OPTIONAL_ARGS+=(--allow-index-wavelengths); fi
if [ "$NMF_SIMPLEX" = "true" ]; then OPTIONAL_ARGS+=(--nmf-simplex); else OPTIONAL_ARGS+=(--no-nmf-simplex); fi
if [ "$AMP" = "true" ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$AUGMENT" = "true" ]; then OPTIONAL_ARGS+=(--augment); else OPTIONAL_ARGS+=(--no-augment); fi
if [ "$EARLY_STOP" = "true" ]; then OPTIONAL_ARGS+=(--early-stop --patience "$PATIENCE"); fi
if [ "$FREEZE_BACKBONE" = "true" ]; then OPTIONAL_ARGS+=(--freeze-backbone); fi

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
    train_finetune_conditioned_cls.py \
    --train-root             "$TRAIN_ROOT" \
    --val-root               "$VAL_ROOT" \
    --test-root              "$TEST_ROOT" \
    --num-classes            $NUM_CLASSES \
    --epochs                 $EPOCHS \
    --batch-size             $BATCH_SIZE_PER_GPU \
    --lr                     $LR \
    --backbone-lr-mult       $BACKBONE_LR_MULT \
    --min-lr                 $MIN_LR \
    --weight-decay           $WEIGHT_DECAY \
    --warmup-epochs          $WARMUP_EPOCHS \
    --clip-grad              $CLIP_GRAD \
    --label-smoothing        $LABEL_SMOOTHING \
    --class-weighting        "$CLASS_WEIGHTING" \
    --sampling-strategy      "$SAMPLING_STRATEGY" \
    --augmentation-copies    $AUGMENTATION_COPIES \
    --augmentation-policy    "$AUGMENTATION_POLICY" \
    --perspective-probability $PERSPECTIVE_PROBABILITY \
    --perspective-scale      $PERSPECTIVE_SCALE \
    --perspective-padding-mode "$PERSPECTIVE_PADDING_MODE" \
    --classification-head    "$CLASSIFICATION_HEAD" \
    --head-projection-dim    $HEAD_PROJECTION_DIM \
    --head-hidden-dim        $HEAD_HIDDEN_DIM \
    --head-dropout           $HEAD_DROPOUT \
    --seed                   $SEED \
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
    --cnn-spectral-agg       "$CNN_SPECTRAL_AGG" \
    --fusion-heads           $FUSION_HEADS \
    --feature-dim            $FEATURE_DIM \
    --decoder-mid-ch         $DECODER_MID_CH \
    --residual-hidden-dim    $RESIDUAL_HIDDEN_DIM \
    --ridge-lambda           $RIDGE_LAMBDA \
    --confidence-temperature $CONFIDENCE_TEMPERATURE \
    --alpha-min              $ALPHA_MIN \
    --alpha-extra            $ALPHA_EXTRA \
    --od-max                 $OD_MAX \
    --workers                $WORKERS \
    --save-interval          $SAVE_INTERVAL \
    --best-val-interval      $BEST_VAL_INTERVAL \
    --progress               "$PROGRESS" \
    --log-interval           $LOG_INTERVAL \
    --save-dir               "$SAVE_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"
