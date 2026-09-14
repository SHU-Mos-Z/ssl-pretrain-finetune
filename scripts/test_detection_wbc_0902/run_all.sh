#!/bin/bash
# Run all WBC detection experiments sequentially with one explicit configuration.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export LR="${LR:-2e-4}"
export EPOCHS="${EPOCHS:-100}"
export WORKERS="${WORKERS:-4}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_train_p075_20260905}"
export VAL_ROOT="${VAL_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_val_p013_20260905}"
export TEST_ROOT="${TEST_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725_finetune_test_p013_20260905}"

EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/00_w_d1_crop640_resize512_fcos.sh"
    "$SCRIPT_DIR/01_w_d2_crop640_resize512_retinanet.sh"
    "$SCRIPT_DIR/02_w_d3_crop512_native_fcos.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    [ -f "$script_path" ] || { echo "Missing experiment script: $script_path"; exit 2; }
done

echo "============================================================================"
echo "WBC detection sequential experiment runner"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU LR=$LR"
echo "EPOCHS=$EPOCHS"
echo "WORKERS=$WORKERS PERSISTENT_WORKERS=$PERSISTENT_WORKERS PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "PRETRAIN_CKPT=$PRETRAIN_CKPT"
echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
echo "============================================================================"

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    echo "========================================================================"
    echo "Starting $(basename "$script_path") on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash "$script_path"
done
