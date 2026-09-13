#!/bin/bash
# Run all GPCC detection experiments sequentially with one explicit configuration.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
export LR="${LR:-8e-4}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_train_p072_20260910}"
export VAL_ROOT="${VAL_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_val_p014_20260910}"
export TEST_ROOT="${TEST_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax_finetune_test_p014_20260910}"

EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/00_gpcc_direct256_fcos_gated.sh"
    "$SCRIPT_DIR/01_gpcc_direct256_retinanet_gated.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    [ -f "$script_path" ] || { echo "Missing experiment script: $script_path"; exit 2; }
done

echo "============================================================================"
echo "GPCC detection sequential experiment runner"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU LR=$LR"
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
