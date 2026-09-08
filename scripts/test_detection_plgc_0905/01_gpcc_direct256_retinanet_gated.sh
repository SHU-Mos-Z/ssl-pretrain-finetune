#!/bin/bash
# GPCC/PLGC-IM D2: preprocessed 256x256 view + gated pyramid + RetinaNet.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "scripts/experiment_naming.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRADIENT_ACCUMULATION_STEPS=1
PYTHON_BIN="${PYTHON_BIN:-python}"

# 固定使用 2026-08-17 的同一份划分；不自动搜索其他日期或比例的数据目录。
TRAIN_ROOT="${TRAIN_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_train_p072_20260817}"
VAL_ROOT="${VAL_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_val_p014_20260817}"
TEST_ROOT="${TEST_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_test_p014_20260817}"
TRAIN_ANNOTATION="annotations"
VAL_ANNOTATION="annotations"
TEST_ANNOTATION="annotations"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"

DETECTION_MODE="anchor_based"
DET_FEATURE_MODE="gated_pyramid"
DETECTION_VIEW_MODE="direct"

EPOCHS=100
LR="${LR:-4e-4}"
BACKBONE_LR_MULT=0.01
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
CLIP_GRAD=1.0
SEED="${SEED:-42}"
AMP=true
AUGMENT=true
FREEZE_BACKBONE=false

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

DET_FEATURE_DIM=128
HEAD_DEPTH=4
POSITIVE_IOU=0.5
NEGATIVE_IOU=0.4
IGNORE_IOU=0.5
BOX_LOSS="giou"
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
SCORE_THRESHOLD=0.05
NMS_THRESHOLD=0.5
PRE_NMS_TOPK=1000
MAX_DETECTIONS=100

# 只依据固定训练集 GT 拟合。256x256 crop 等于整张模型输入，因此拟合过程中
# 不会产生额外的运行时裁剪；dry-run 会在该步骤之前退出。
ANCHOR_CONFIG_JSON="records/test_detection_plgc_0905/anchor_fit_GPCC_train_p072_direct256_seed${SEED}.json"
ANCHOR_SOURCE_CROP_SIZE="256,256"
ANCHOR_MODEL_INPUT_SIZE="256,256"

NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
NMF_CACHE_NAME="nmf_cache_K16_l15e-4_l22e-4_l31e-2_le0.05_ec3_simplex"
ALLOW_INDEX_WAVELENGTHS=true
WAVELENGTH_FILE=""

WORKERS=2
SAVE_INTERVAL=10
EVALUATION_INTERVAL=5
PR_CURVE_INTERVAL=5
PROGRESS="log"
LOG_INTERVAL=10

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    for required in images masks ignore_masks annotations "$NMF_CACHE_NAME"; do
        if [ ! -d "$root/$required" ]; then
            echo "Missing required GPCC detection input: $root/$required" >&2
            exit 1
        fi
    done
    if [ ! -f "$root/wavelengths.npy" ]; then
        echo "Missing wavelength metadata: $root/wavelengths.npy" >&2
        exit 1
    fi
done
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing pretraining checkpoint: $PRETRAIN_CKPT" >&2
    exit 1
fi

echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
echo "DETECTION_VIEW_MODE=$DETECTION_VIEW_MODE  DETECTION_MODE=$DETECTION_MODE"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: fixed split paths, annotations, NMF caches and checkpoint passed validation; anchor fitting and training were not started."
    exit 0
fi

mkdir -p "$(dirname "$ANCHOR_CONFIG_JSON")"
if [ ! -f "$ANCHOR_CONFIG_JSON" ]; then
    "$PYTHON_BIN" scripts/fit_wbc_detection_anchors.py \
        --data-root "$TRAIN_ROOT" --annotation "$TRAIN_ANNOTATION" \
        --source-crop-size "$ANCHOR_SOURCE_CROP_SIZE" \
        --model-input-size "$ANCHOR_MODEL_INPUT_SIZE" \
        --views-per-source 1 --simulation-epochs 1 \
        --positive-guided-fraction 1.0 --visible-ratio 0.0 --min-visible-side 0.0 \
        --seed "$SEED" --output "$ANCHOR_CONFIG_JSON"
fi

DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")-direct256-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="records/test_detection_plgc_0905/GPCC-D2_${DATASET_INFO}_retinanet_gated_seed${SEED}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
echo "ANCHOR_CONFIG_JSON=$ANCHOR_CONFIG_JSON"
echo "SAVE_DIR=$SAVE_DIR"

OPTIONAL_ARGS=()
if [ "$AMP" = true ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$AUGMENT" = true ]; then OPTIONAL_ARGS+=(--augment); else OPTIONAL_ARGS+=(--no-augment); fi
if [ "$FREEZE_BACKBONE" = true ]; then OPTIONAL_ARGS+=(--freeze-backbone); fi
if [ "$NMF_SIMPLEX" = true ]; then OPTIONAL_ARGS+=(--nmf-simplex); else OPTIONAL_ARGS+=(--no-nmf-simplex); fi
if [ "$ALLOW_INDEX_WAVELENGTHS" = true ]; then OPTIONAL_ARGS+=(--allow-index-wavelengths); fi
if [ -n "$WAVELENGTH_FILE" ]; then OPTIONAL_ARGS+=(--wavelength-file "$WAVELENGTH_FILE"); fi

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train_finetune_conditioned_detection.py \
    --train-root "$TRAIN_ROOT" --train-annotation "$TRAIN_ANNOTATION" \
    --val-root "$VAL_ROOT" --val-annotation "$VAL_ANNOTATION" \
    --test-root "$TEST_ROOT" --test-annotation "$TEST_ANNOTATION" \
    --pretrain-ckpt "$PRETRAIN_CKPT" \
    --detection-mode "$DETECTION_MODE" --det-feature-mode "$DET_FEATURE_MODE" \
    --detection-view-mode "$DETECTION_VIEW_MODE" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE_PER_GPU" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --backbone-lr-mult "$BACKBONE_LR_MULT" --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS" \
    --clip-grad "$CLIP_GRAD" --workers "$WORKERS" --seed "$SEED" \
    --save-interval "$SAVE_INTERVAL" --evaluation-interval "$EVALUATION_INTERVAL" \
    --pr-curve-interval "$PR_CURVE_INTERVAL" --progress "$PROGRESS" --log-interval "$LOG_INTERVAL" \
    --patch-size "$PATCH_SIZE" --spectral-patch-size "$SPECTRAL_PATCH_SIZE" \
    --embed-dim "$EMBED_DIM" --vit-depth "$VIT_DEPTH" --vit-heads "$VIT_HEADS" \
    --mlp-ratio "$MLP_RATIO" --dropout "$DROPOUT" --cnn-stem-ch "$CNN_STEM_CH" \
    --cnn-spectral-agg "$CNN_SPECTRAL_AGG" --fusion-heads "$FUSION_HEADS" \
    --feature-dim "$FEATURE_DIM" --decoder-mid-ch "$DECODER_MID_CH" \
    --residual-hidden-dim "$RESIDUAL_HIDDEN_DIM" --ridge-lambda "$RIDGE_LAMBDA" \
    --confidence-temperature "$CONFIDENCE_TEMPERATURE" --alpha-min "$ALPHA_MIN" \
    --alpha-extra "$ALPHA_EXTRA" --od-max "$OD_MAX" \
    --det-feature-dim "$DET_FEATURE_DIM" --head-depth "$HEAD_DEPTH" \
    --anchor-config-json "$ANCHOR_CONFIG_JSON" \
    --positive-iou-threshold "$POSITIVE_IOU" --negative-iou-threshold "$NEGATIVE_IOU" \
    --ignore-iou-threshold "$IGNORE_IOU" --box-loss "$BOX_LOSS" \
    --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" \
    --score-threshold "$SCORE_THRESHOLD" --nms-threshold "$NMS_THRESHOLD" \
    --pre-nms-topk "$PRE_NMS_TOPK" --max-detections "$MAX_DETECTIONS" \
    --nmf-k "$NMF_K" --nmf-l1 "$NMF_L1" --nmf-l2 "$NMF_L2" --nmf-l3 "$NMF_L3" \
    --nmf-lam-e "$NMF_LAM_E" --nmf-e-clamp-max "$NMF_E_CLAMP_MAX" \
    --save-dir "$SAVE_DIR" "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"
