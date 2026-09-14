#!/bin/bash
# ViT backbone 分割微调
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

TRAIN_ROOT="${TRAIN_ROOT:-data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_train}"
VAL_ROOT="${VAL_ROOT:-data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_val}"
TEST_ROOT="${TEST_ROOT:-data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_test}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-./records/pretrain/20260630_094215/ckpt_epoch0200.pth}"   # 例如 records/pretrain/20260629_120000/ckpt_epoch0200.pth

EPOCHS=100
LR="${LR:-5e-4}"
MIN_LR=1e-6
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5
CLIP_GRAD=1.0
SEED=42
EARLY_STOP=false
PATIENCE=20

NUM_CLASSES=2
EMBED_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
PATCH_SIZE=16
SPECTRAL_PATCH_SIZE=10
NUM_ENDMEMBERS=8
AGGREGATE_MODE="mean"
FREEZE_BACKBONE=false
# 留空时不传参，由 Python 使用历史 all_class 损失；设为 foreground
# 时使用全类别加权 CE + 前景 Dice + 前景 boundary。
SEGMENTATION_LOSS_MODE="${SEGMENTATION_LOSS_MODE:-}"

WORKERS=4
SAVE_INTERVAL=10
N_VIS=8
PROGRESS="log"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/finetune/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

PRETRAIN_ARG=""
if [ -n "$PRETRAIN_CKPT" ]; then
    PRETRAIN_ARG="--pretrain-ckpt $PRETRAIN_CKPT"
fi
EARLY_STOP_ARG=""
if [ "$EARLY_STOP" = "true" ]; then
    EARLY_STOP_ARG="--early-stop --patience $PATIENCE"
fi
FREEZE_ARG=""
if [ "$FREEZE_BACKBONE" = "true" ]; then
    FREEZE_ARG="--freeze-backbone"
fi
SEGMENTATION_LOSS_MODE_ARGS=()
if [ -n "$SEGMENTATION_LOSS_MODE" ]; then
    SEGMENTATION_LOSS_MODE_ARGS+=(--segmentation-loss-mode "$SEGMENTATION_LOSS_MODE")
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
    train_finetune_vit.py \
    --train-root         "$TRAIN_ROOT" \
    --val-root           "$VAL_ROOT" \
    --test-root          "$TEST_ROOT" \
    --epochs             $EPOCHS \
    --batch-size         $BATCH_SIZE_PER_GPU \
    --lr                 $LR \
    --min-lr             $MIN_LR \
    --weight-decay       $WEIGHT_DECAY \
    --warmup-epochs      $WARMUP_EPOCHS \
    --clip-grad          $CLIP_GRAD \
    --seed               $SEED \
    --num-classes        $NUM_CLASSES \
    --embed-dim          $EMBED_DIM \
    --vit-depth          $VIT_DEPTH \
    --vit-heads          $VIT_HEADS \
    --patch-size         $PATCH_SIZE \
    --spectral-patch-size $SPECTRAL_PATCH_SIZE \
    --num-endmembers     $NUM_ENDMEMBERS \
    --aggregate-mode     $AGGREGATE_MODE \
    "${SEGMENTATION_LOSS_MODE_ARGS[@]}" \
    --workers            $WORKERS \
    --save-interval      $SAVE_INTERVAL \
    --n-vis              $N_VIS \
    --progress           $PROGRESS \
    --save-dir           "$SAVE_DIR" \
    --amp \
    $PRETRAIN_ARG \
    $EARLY_STOP_ARG \
    $FREEZE_ARG \
    2>&1 | tee "$SAVE_DIR/records.txt"
