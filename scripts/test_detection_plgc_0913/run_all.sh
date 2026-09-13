#!/bin/bash
# Run all eight GPCC detector/feature/head combinations sequentially.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Shared experiment controls. Override these when switching from 256x320 to
# native 512x640 data; every child script inherits the exact same protocol.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export LR="${LR:-4e-4}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_20260913_finetune_train_p072_20260913}"
export VAL_ROOT="${VAL_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_20260913_finetune_val_p014_20260913}"
export TEST_ROOT="${TEST_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_20260913_finetune_test_p014_20260913}"

export EPOCHS="${EPOCHS:-100}"
export SEED="${SEED:-42}"
export AP_SCORE_THRESHOLD="${AP_SCORE_THRESHOLD:-0.05}"
export DEPLOY_SCORE_THRESHOLD="${DEPLOY_SCORE_THRESHOLD:-}"
export VIS_SCORE_THRESHOLD="${VIS_SCORE_THRESHOLD:-}"
export VIS_MAX_DETECTIONS="${VIS_MAX_DETECTIONS:-30}"
export THRESHOLD_CALIBRATION_IOU="${THRESHOLD_CALIBRATION_IOU:-0.5}"
export THRESHOLD_SEARCH_MIN="${THRESHOLD_SEARCH_MIN:-0.05}"
export THRESHOLD_SEARCH_MAX="${THRESHOLD_SEARCH_MAX:-0.90}"
export THRESHOLD_SEARCH_STEP="${THRESHOLD_SEARCH_STEP:-0.01}"
export TEST_VISUALIZATION_SAMPLES="${TEST_VISUALIZATION_SAMPLES:-12}"
export PROGRESS="${PROGRESS:-log}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"

EXPERIMENT_SCRIPTS=(
    # "$SCRIPT_DIR/00_gpcc_fcos_gated_pyramid_legacy.sh"
    # "$SCRIPT_DIR/01_gpcc_fcos_gated_pyramid_gn_quality.sh"
    # "$SCRIPT_DIR/02_gpcc_fcos_gated_fpn_legacy.sh"
    # "$SCRIPT_DIR/03_gpcc_fcos_gated_fpn_gn_quality.sh"
    "$SCRIPT_DIR/04_gpcc_retinanet_gated_pyramid_legacy.sh"
    "$SCRIPT_DIR/05_gpcc_retinanet_gated_pyramid_gn_quality.sh"
    "$SCRIPT_DIR/06_gpcc_retinanet_gated_fpn_legacy.sh"
    "$SCRIPT_DIR/07_gpcc_retinanet_gated_fpn_gn_quality.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    [ -f "$script_path" ] || { echo "Missing experiment script: $script_path" >&2; exit 2; }
done

echo "============================================================================"
echo "GPCC 0913 sequential detection experiments"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS"
echo "BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU ACCUMULATION=$GRADIENT_ACCUMULATION_STEPS LR=$LR"
echo "PRETRAIN_CKPT=$PRETRAIN_CKPT"
echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
echo "AP_SCORE_THRESHOLD=$AP_SCORE_THRESHOLD DEPLOY_SCORE_THRESHOLD=${DEPLOY_SCORE_THRESHOLD:-auto}"
echo "============================================================================"

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    echo "========================================================================"
    echo "Starting $(basename "$script_path") on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash "$script_path"
done
