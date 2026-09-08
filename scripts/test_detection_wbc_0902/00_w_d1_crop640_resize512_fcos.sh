#!/bin/bash
# W-D1: 640x640 source window -> 512x512 input, gated pyramid + FCOS.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "scripts/experiment_naming.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS=2

# 本实验固定使用 2026-09-05 划分；禁止按日期或比例自动发现其他目录。
TRAIN_ROOT="${TRAIN_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_train_p075_20260905}"
VAL_ROOT="${VAL_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_val_p013_20260905}"
TEST_ROOT="${TEST_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_test_p013_20260905}"
TRAIN_ANNOTATION="annotations"
VAL_ANNOTATION="annotations"
TEST_ANNOTATION="annotations"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"

DETECTION_MODE="anchor_free"
DET_FEATURE_MODE="gated_pyramid"
DETECTION_VIEW_MODE="runtime_window"
SOURCE_CROP_SIZE="640,640"
MODEL_INPUT_SIZE="512,512"
TRAIN_VIEWS_PER_SOURCE=4
POSITIVE_GUIDED_FRACTION=0.5
EVAL_STRIDE="224,224"
RUNTIME_VISIBLE_RATIO=0.70
RUNTIME_MIN_VISIBLE_SIDE=32
GLOBAL_NMS_THRESHOLD=0.5

EPOCHS=100
LR="${LR:-2e-4}"
BACKBONE_LR_MULT=0.1
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
CLIP_GRAD=1.0
SEED="${SEED:-42}"
WORKERS=2
PR_CURVE_INTERVAL=5
EVALUATION_INTERVAL=5
PROGRESS="log"
LOG_INTERVAL=10

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
ANCHOR_SIZES="32,64,128,256"
ANCHOR_SCALES="0.8,1.0,1.25"
ANCHOR_RATIOS="0.67,1.0,1.5"
POSITIVE_IOU=0.5
NEGATIVE_IOU=0.4
IGNORE_IOU=0.5
BOX_LOSS="giou"
FCOS_RANGES="0:32,32:64,64:128,128:100000000"
FCOS_CENTER_RADIUS=1.5
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
SCORE_THRESHOLD=0.05
NMS_THRESHOLD=0.5
PRE_NMS_TOPK=1000
MAX_DETECTIONS=100

NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ -z "$root" ] || [ ! -d "$root" ]; then
        echo "Missing WBC detection split. Run scripts/run_split_pretrain_finetune_detection.sh first."
        exit 1
    fi
    [ -d "$root/annotations" ] || { echo "Missing $root/annotations"; exit 1; }
done
[ -f "$PRETRAIN_CKPT" ] || { echo "Missing $PRETRAIN_CKPT"; exit 1; }

echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: fixed split paths and required inputs passed validation; training was not started."
    exit 0
fi

DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")-crop640x640-in512x512-s224x224-tok16-sp5"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="records/test_detection_wbc_0902/W-D1_${DATASET_INFO}_fcos_gated_seed${SEED}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train_finetune_conditioned_detection.py \
    --train-root "$TRAIN_ROOT" --train-annotation "$TRAIN_ANNOTATION" \
    --val-root "$VAL_ROOT" --val-annotation "$VAL_ANNOTATION" \
    --test-root "$TEST_ROOT" --test-annotation "$TEST_ANNOTATION" \
    --pretrain-ckpt "$PRETRAIN_CKPT" \
    --detection-mode "$DETECTION_MODE" --det-feature-mode "$DET_FEATURE_MODE" \
    --detection-view-mode "$DETECTION_VIEW_MODE" \
    --source-crop-size "$SOURCE_CROP_SIZE" --model-input-size "$MODEL_INPUT_SIZE" \
    --train-views-per-source "$TRAIN_VIEWS_PER_SOURCE" \
    --positive-guided-fraction "$POSITIVE_GUIDED_FRACTION" --eval-stride "$EVAL_STRIDE" \
    --runtime-visible-ratio "$RUNTIME_VISIBLE_RATIO" \
    --runtime-min-visible-side "$RUNTIME_MIN_VISIBLE_SIDE" \
    --enable-crop-truncated-positive --eval-ownership-filter \
    --global-nms-threshold "$GLOBAL_NMS_THRESHOLD" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE_PER_GPU" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --backbone-lr-mult "$BACKBONE_LR_MULT" --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS" \
    --clip-grad "$CLIP_GRAD" --workers "$WORKERS" --seed "$SEED" --amp --augment \
    --pr-curve-interval "$PR_CURVE_INTERVAL" --evaluation-interval "$EVALUATION_INTERVAL" \
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
    --anchor-sizes "$ANCHOR_SIZES" --anchor-scales "$ANCHOR_SCALES" --anchor-ratios "$ANCHOR_RATIOS" \
    --positive-iou-threshold "$POSITIVE_IOU" --negative-iou-threshold "$NEGATIVE_IOU" \
    --ignore-iou-threshold "$IGNORE_IOU" --box-loss "$BOX_LOSS" \
    --fcos-regression-ranges "$FCOS_RANGES" --fcos-center-sampling-radius "$FCOS_CENTER_RADIUS" \
    --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" \
    --score-threshold "$SCORE_THRESHOLD" --nms-threshold "$NMS_THRESHOLD" \
    --pre-nms-topk "$PRE_NMS_TOPK" --max-detections "$MAX_DETECTIONS" \
    --nmf-k "$NMF_K" --nmf-l1 "$NMF_L1" --nmf-l2 "$NMF_L2" --nmf-l3 "$NMF_L3" \
    --nmf-simplex --nmf-lam-e "$NMF_LAM_E" --nmf-e-clamp-max "$NMF_E_CLAMP_MAX" \
    --allow-index-wavelengths --save-dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/records.txt"
