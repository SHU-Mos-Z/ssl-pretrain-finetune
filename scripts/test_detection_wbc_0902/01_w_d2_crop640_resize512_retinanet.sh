#!/bin/bash
# W-D2: exactly W-D1's views, gated pyramid + RetinaNet with train-only fitted anchors.
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
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"

DETECTION_MODE="anchor_based"
DET_FEATURE_MODE="gated_pyramid"
SOURCE_CROP_SIZE="640,640"
MODEL_INPUT_SIZE="512,512"
EVAL_STRIDE="224,224"
TRAIN_VIEWS_PER_SOURCE=4
POSITIVE_GUIDED_FRACTION=0.5
RUNTIME_VISIBLE_RATIO=0.70
RUNTIME_MIN_VISIBLE_SIDE=32
GLOBAL_NMS_THRESHOLD=0.5

EPOCHS=100
LR="${LR:-2e-4}"
BACKBONE_LR_MULT=0.1
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
SEED="${SEED:-42}"
WORKERS=2
ANCHOR_CONFIG_JSON="records/test_detection_wbc_0902/anchor_fit_$(basename "$TRAIN_ROOT")_crop640_in512_seed${SEED}.json"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ -z "$root" ] || [ ! -d "$root/annotations" ]; then
        echo "Missing WBC detection split. Run scripts/run_split_pretrain_finetune_detection.sh first."
        exit 1
    fi
done
[ -f "$PRETRAIN_CKPT" ] || { echo "Missing $PRETRAIN_CKPT"; exit 1; }
echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: fixed split paths and required inputs passed validation; anchor fitting and training were not started."
    exit 0
fi
mkdir -p "$(dirname "$ANCHOR_CONFIG_JSON")"
if [ ! -f "$ANCHOR_CONFIG_JSON" ]; then
    conda run -n zsq_accl_mine python scripts/fit_wbc_detection_anchors.py \
        --data-root "$TRAIN_ROOT" --annotation annotations \
        --source-crop-size "$SOURCE_CROP_SIZE" --model-input-size "$MODEL_INPUT_SIZE" \
        --views-per-source "$TRAIN_VIEWS_PER_SOURCE" --simulation-epochs 10 \
        --positive-guided-fraction "$POSITIVE_GUIDED_FRACTION" \
        --visible-ratio "$RUNTIME_VISIBLE_RATIO" --min-visible-side "$RUNTIME_MIN_VISIBLE_SIDE" \
        --seed "$SEED" --output "$ANCHOR_CONFIG_JSON"
fi

DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")-crop640x640-in512x512-s224x224-tok16-sp5"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="records/test_detection_wbc_0902/W-D2_${DATASET_INFO}_retinanet_gated_seed${SEED}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train_finetune_conditioned_detection.py \
    --train-root "$TRAIN_ROOT" --train-annotation annotations \
    --val-root "$VAL_ROOT" --val-annotation annotations \
    --test-root "$TEST_ROOT" --test-annotation annotations \
    --pretrain-ckpt "$PRETRAIN_CKPT" \
    --detection-mode "$DETECTION_MODE" --det-feature-mode "$DET_FEATURE_MODE" \
    --detection-view-mode runtime_window --source-crop-size "$SOURCE_CROP_SIZE" \
    --model-input-size "$MODEL_INPUT_SIZE" --eval-stride "$EVAL_STRIDE" \
    --train-views-per-source "$TRAIN_VIEWS_PER_SOURCE" \
    --positive-guided-fraction "$POSITIVE_GUIDED_FRACTION" \
    --runtime-visible-ratio "$RUNTIME_VISIBLE_RATIO" \
    --runtime-min-visible-side "$RUNTIME_MIN_VISIBLE_SIDE" \
    --enable-crop-truncated-positive --eval-ownership-filter \
    --global-nms-threshold "$GLOBAL_NMS_THRESHOLD" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE_PER_GPU" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --backbone-lr-mult "$BACKBONE_LR_MULT" --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS" --clip-grad 1.0 \
    --workers "$WORKERS" --seed "$SEED" --amp --augment \
    --pr-curve-interval 5 --evaluation-interval 5 --progress log --log-interval 10 \
    --patch-size 16 --spectral-patch-size 5 --embed-dim 256 --vit-depth 6 --vit-heads 8 \
    --mlp-ratio 4.0 --dropout 0.1 --cnn-stem-ch 64 --cnn-spectral-agg attention \
    --fusion-heads 8 --feature-dim 128 --decoder-mid-ch 64 --residual-hidden-dim 128 \
    --ridge-lambda 1e-3 --confidence-temperature 0.05 --alpha-min 0.1 --alpha-extra 1.0 --od-max 3.0 \
    --det-feature-dim 128 --head-depth 4 --anchor-config-json "$ANCHOR_CONFIG_JSON" \
    --positive-iou-threshold 0.5 --negative-iou-threshold 0.4 --ignore-iou-threshold 0.5 \
    --box-loss giou --focal-alpha 0.25 --focal-gamma 2.0 \
    --score-threshold 0.05 --nms-threshold 0.5 --pre-nms-topk 1000 --max-detections 100 \
    --nmf-k 16 --nmf-l1 5e-4 --nmf-l2 2e-4 --nmf-l3 1e-2 \
    --nmf-simplex --nmf-lam-e 0.05 --nmf-e-clamp-max 3.0 --allow-index-wavelengths \
    --save-dir "$SAVE_DIR" 2>&1 | tee "$SAVE_DIR/records.txt"
