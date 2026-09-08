#!/bin/bash
# Experiment 3/7: anchor-based detector, native gated-decoder pyramid, frozen pretrained backbone.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "scripts/experiment_naming.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# ── 核心实验开关：支持全部六种组合 ───────────────────────────────────────────
DETECTION_MODE="anchor_based"   # anchor_based | anchor_free
DET_FEATURE_MODE="gated_pyramid"    # z_pyramid | gated_pyramid | z_full

# train/val/test 是同一检测数据集经 run_split_pretrain_finetune.sh 得到的三个
# split 根目录；“out-of-domain”只由预训练数据集是否不同决定。
# 下列目录来自 2026-08-17 已完成的检测数据划分。

# MDC（检测）
TRAIN_ROOT="${TRAIN_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x256_xml_object_minmax_finetune_train_p070_20260817}"
VAL_ROOT="${VAL_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x256_xml_object_minmax_finetune_val_p015_20260817}"
TEST_ROOT="${TEST_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x256_xml_object_minmax_finetune_test_p015_20260817}"

# GPCC（检测）
# TRAIN_ROOT="data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_train_p072_20260817"
# VAL_ROOT="data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_val_p014_20260817"
# TEST_ROOT="data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_test_p014_20260817"

# 可以指向逐 Patch JSON 目录（推荐），也继续兼容单个 COCO JSON 文件。
TRAIN_ANNOTATION="annotations"
VAL_ANNOTATION="annotations"
TEST_ANNOTATION="annotations"

# Conditioned 预训练 checkpoint；留空时为完全监督 scratch baseline。
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"

# ── 训练超参数 ────────────────────────────────────────────────────────────────
EPOCHS=100
LR="${LR:-4e-4}"
BACKBONE_LR_MULT=0.1
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
CLIP_GRAD=1.0
SEED=42
AMP=true
AUGMENT=true
FREEZE_BACKBONE=true
EXPERIMENT_TAG="frozen"

# ── 模型架构（须与预训练保持一致） ───────────────────────────────────────────
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

# ── 检测几何与损失 ────────────────────────────────────────────────────────────
# 最终 anchor 模板只能依据训练集 GT 拟合。
DET_FEATURE_DIM=128
HEAD_DEPTH=4
ANCHOR_SIZES="16,32,64,128"
ANCHOR_SCALES="1.0,1.2599,1.5874"
ANCHOR_RATIOS="0.5,1.0,2.0"
POSITIVE_IOU=0.5
NEGATIVE_IOU=0.4
IGNORE_IOU=0.5
BOX_LOSS="smooth_l1"
FCOS_RANGES="0:32,32:64,64:128,128:100000000"
FCOS_CENTER_RADIUS=1.5
FCOS_NORMALIZE_REG_TARGETS_BY_STRIDE=true
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
SCORE_THRESHOLD=0.05
NMS_THRESHOLD=0.5
PRE_NMS_TOPK=1000
MAX_DETECTIONS=100

# ── NMF 条件输入缓存键（须与预训练保持一致） ─────────────────────────────────
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
ALLOW_INDEX_WAVELENGTHS=true
WAVELENGTH_FILE=""

# ── 日志 / 存储 ────────────────────────────────────────────────────────────────
WORKERS=2
PR_CURVE_INTERVAL=5       # 每 N 个 epoch 保存一次验证集 PR 曲线；<=0 时禁用
PROGRESS="log"
LOG_INTERVAL=10
DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/test_detection_0817/${DATASET_INFO}_${DETECTION_MODE}_${DET_FEATURE_MODE}_${EXPERIMENT_TAG}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
echo "DATASET_INFO=$DATASET_INFO"
echo "SAVE_DIR=$SAVE_DIR"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ ! -d "$root" ]; then
        echo "Missing detection split root: $root"
        echo "Run scripts/run_split_pretrain_finetune.sh with KIND=detection first."
        exit 1
    fi
done
for spec in \
    "$TRAIN_ROOT|$TRAIN_ANNOTATION" \
    "$VAL_ROOT|$VAL_ANNOTATION" \
    "$TEST_ROOT|$TEST_ANNOTATION"; do
    root="${spec%%|*}"
    annotation="${spec#*|}"
    annotation_source="$annotation"
    if [[ "$annotation_source" != /* ]]; then
        annotation_source="$root/$annotation_source"
    fi
    if [ ! -e "$annotation_source" ]; then
        echo "Missing detection annotation source: $annotation_source"
        exit 1
    fi
done
if [ -n "$PRETRAIN_CKPT" ] && [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing pretraining checkpoint: $PRETRAIN_CKPT"
    exit 1
fi

OPTIONAL_ARGS=()
if [ -n "$PRETRAIN_CKPT" ]; then OPTIONAL_ARGS+=(--pretrain-ckpt "$PRETRAIN_CKPT"); fi
if [ -n "$WAVELENGTH_FILE" ]; then OPTIONAL_ARGS+=(--wavelength-file "$WAVELENGTH_FILE"); fi
if [ "$ALLOW_INDEX_WAVELENGTHS" = true ]; then OPTIONAL_ARGS+=(--allow-index-wavelengths); fi
if [ "$NMF_SIMPLEX" = true ]; then OPTIONAL_ARGS+=(--nmf-simplex); else OPTIONAL_ARGS+=(--no-nmf-simplex); fi
if [ "$FCOS_NORMALIZE_REG_TARGETS_BY_STRIDE" = true ]; then
    OPTIONAL_ARGS+=(--fcos-normalize-reg-targets-by-stride)
else
    OPTIONAL_ARGS+=(--no-fcos-normalize-reg-targets-by-stride)
fi
if [ "$AMP" = true ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$AUGMENT" = true ]; then OPTIONAL_ARGS+=(--augment); else OPTIONAL_ARGS+=(--no-augment); fi
if [ "$FREEZE_BACKBONE" = true ]; then OPTIONAL_ARGS+=(--freeze-backbone); fi

echo "TRAIN_ROOT=$TRAIN_ROOT  annotation=$TRAIN_ANNOTATION"
echo "VAL_ROOT=$VAL_ROOT  annotation=$VAL_ANNOTATION"
echo "TEST_ROOT=$TEST_ROOT  annotation=$TEST_ANNOTATION"

MASTER_PORT=$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
")

OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train_finetune_conditioned_detection.py \
    --train-root "$TRAIN_ROOT" --train-annotation "$TRAIN_ANNOTATION" \
    --val-root "$VAL_ROOT" --val-annotation "$VAL_ANNOTATION" \
    --test-root "$TEST_ROOT" --test-annotation "$TEST_ANNOTATION" \
    --detection-mode "$DETECTION_MODE" --det-feature-mode "$DET_FEATURE_MODE" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE_PER_GPU" --lr "$LR" \
    --backbone-lr-mult "$BACKBONE_LR_MULT" --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS" \
    --clip-grad "$CLIP_GRAD" --workers "$WORKERS" --seed "$SEED" \
    --pr-curve-interval "$PR_CURVE_INTERVAL" \
    --progress "$PROGRESS" --log-interval "$LOG_INTERVAL" \
    --patch-size "$PATCH_SIZE" --spectral-patch-size "$SPECTRAL_PATCH_SIZE" \
    --embed-dim "$EMBED_DIM" --vit-depth "$VIT_DEPTH" --vit-heads "$VIT_HEADS" \
    --mlp-ratio "$MLP_RATIO" --dropout "$DROPOUT" --cnn-stem-ch "$CNN_STEM_CH" \
    --cnn-spectral-agg "$CNN_SPECTRAL_AGG" --fusion-heads "$FUSION_HEADS" \
    --feature-dim "$FEATURE_DIM" --decoder-mid-ch "$DECODER_MID_CH" \
    --residual-hidden-dim "$RESIDUAL_HIDDEN_DIM" --ridge-lambda "$RIDGE_LAMBDA" \
    --confidence-temperature "$CONFIDENCE_TEMPERATURE" --alpha-min "$ALPHA_MIN" \
    --alpha-extra "$ALPHA_EXTRA" --od-max "$OD_MAX" \
    --det-feature-dim "$DET_FEATURE_DIM" --head-depth "$HEAD_DEPTH" \
    --anchor-sizes "$ANCHOR_SIZES" --anchor-scales "$ANCHOR_SCALES" \
    --anchor-ratios "$ANCHOR_RATIOS" --positive-iou-threshold "$POSITIVE_IOU" \
    --negative-iou-threshold "$NEGATIVE_IOU" --ignore-iou-threshold "$IGNORE_IOU" \
    --box-loss "$BOX_LOSS" --fcos-regression-ranges "$FCOS_RANGES" \
    --fcos-center-sampling-radius "$FCOS_CENTER_RADIUS" \
    --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" \
    --score-threshold "$SCORE_THRESHOLD" --nms-threshold "$NMS_THRESHOLD" \
    --pre-nms-topk "$PRE_NMS_TOPK" --max-detections "$MAX_DETECTIONS" \
    --nmf-k "$NMF_K" --nmf-l1 "$NMF_L1" --nmf-l2 "$NMF_L2" --nmf-l3 "$NMF_L3" \
    --nmf-lam-e "$NMF_LAM_E" --nmf-e-clamp-max "$NMF_E_CLAMP_MAX" \
    --save-dir "$SAVE_DIR" "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"
