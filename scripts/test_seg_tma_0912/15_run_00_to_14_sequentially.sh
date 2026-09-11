#!/bin/bash
# Run the TMA segmentation experiment matrix sequentially on one fixed GPU set.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# These values are exported once so every child experiment uses the same
# hardware, effective batch size, learning rate, checkpoint, and fixed splits.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
export LR="${LR:-4e-4}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/TMA_patch_1024x1024_overlap_0x0_to_256x256_minmax_finetune_train_p070_20260910}"
export VAL_ROOT="${VAL_ROOT:-data/TMA_patch_1024x1024_overlap_0x0_to_256x256_minmax_finetune_val_p015_20260910}"
export TEST_ROOT="${TEST_ROOT:-data/TMA_patch_1024x1024_overlap_0x0_to_256x256_minmax_finetune_test_p015_20260910}"

# Data-pipeline controls; these do not alter model or optimization semantics.
export WORKERS="${WORKERS:-2}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export DISTRIBUTED_VALIDATION="${DISTRIBUTED_VALIDATION:-true}"

# H/L/O stages need a fixed predecessor choice when the full matrix is run in
# one unattended job. Override these after inspecting earlier-stage results.
export BEST_AUGMENTATION_POLICY="${BEST_AUGMENTATION_POLICY:-dihedral}"
export SCREEN_SEGMENTATION_HEAD="${SCREEN_SEGMENTATION_HEAD:-h0_simple}"
export SCREEN_AUX_LOSS_WEIGHT="${SCREEN_AUX_LOSS_WEIGHT:-0.0}"
export SCREEN_SEGMENTATION_LOSS="${SCREEN_SEGMENTATION_LOSS:-ce_dice}"
export SCREEN_CLASS_WEIGHT_MODE="${SCREEN_CLASS_WEIGHT_MODE:-none}"
export SCREEN_BOUNDARY_LOSS_WEIGHT="${SCREEN_BOUNDARY_LOSS_WEIGHT:-0.0}"
export SCREEN_FOCAL_GAMMA="${SCREEN_FOCAL_GAMMA:-2.0}"

# Default final combination is a reasonable prior only. For the formal
# multi-seed run, override FINAL_* with the settings selected on validation.
export FINAL_AUGMENTATION_COPIES="${FINAL_AUGMENTATION_COPIES:-1}"
export FINAL_AUGMENTATION_POLICY="${FINAL_AUGMENTATION_POLICY:-dihedral_affine}"
export FINAL_SEGMENTATION_HEAD="${FINAL_SEGMENTATION_HEAD:-h3_multiscale_aux}"
export FINAL_AUX_LOSS_WEIGHT="${FINAL_AUX_LOSS_WEIGHT:-0.4}"
export FINAL_SEGMENTATION_LOSS="${FINAL_SEGMENTATION_LOSS:-weighted_ce_dice_boundary}"
export FINAL_CLASS_WEIGHT_MODE="${FINAL_CLASS_WEIGHT_MODE:-inverse_sqrt}"
export FINAL_BOUNDARY_LOSS_WEIGHT="${FINAL_BOUNDARY_LOSS_WEIGHT:-0.2}"
export FINAL_BACKBONE_LR_MULTIPLIER="${FINAL_BACKBONE_LR_MULTIPLIER:-0.1}"

EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/00_t_00_default.sh"
    "$SCRIPT_DIR/01_t_a1_dihedral.sh"
    "$SCRIPT_DIR/02_t_a2_dihedral_affine.sh"
    "$SCRIPT_DIR/03_t_a3_dihedral_perspective.sh"
    "$SCRIPT_DIR/04_t_h1_residual.sh"
    "$SCRIPT_DIR/05_t_h2_aspp.sh"
    "$SCRIPT_DIR/06_t_h3_multiscale_aux.sh"
    "$SCRIPT_DIR/07_t_l1_weighted_ce_dice.sh"
    "$SCRIPT_DIR/08_t_l2_focal_dice.sh"
    "$SCRIPT_DIR/09_t_l3_weighted_boundary.sh"
    "$SCRIPT_DIR/10_t_o1_discriminative_lr.sh"
    "$SCRIPT_DIR/11_t_o2_staged_unfreeze.sh"
    "$SCRIPT_DIR/12_t_final_seed42.sh"
    "$SCRIPT_DIR/13_t_final_seed43.sh"
    "$SCRIPT_DIR/14_t_final_seed44.sh"
)

for required_path in \
    "$PRETRAIN_CKPT" "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    [ -e "$required_path" ] || {
        echo "Missing required path: $required_path" >&2
        exit 2
    }
done
for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    [ -f "$script_path" ] || {
        echo "Missing experiment script: $script_path" >&2
        exit 2
    }
done

echo "============================================================================"
echo "TMA sequential segmentation experiment runner"
echo "Script directory:       $SCRIPT_DIR"
echo "Project root:           $PROJECT_ROOT"
echo "CUDA_VISIBLE_DEVICES:   $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS:               $NUM_GPUS"
echo "BATCH_SIZE_PER_GPU:     $BATCH_SIZE_PER_GPU"
echo "LR:                     $LR"
echo "PRETRAIN_CKPT:          $PRETRAIN_CKPT"
echo "TRAIN_ROOT:             $TRAIN_ROOT"
echo "VAL_ROOT:               $VAL_ROOT"
echo "TEST_ROOT:              $TEST_ROOT"
echo "BEST_AUGMENTATION:      $BEST_AUGMENTATION_POLICY"
echo "SCREEN_HEAD:            $SCREEN_SEGMENTATION_HEAD"
echo "SCREEN_AUX_WEIGHT:      $SCREEN_AUX_LOSS_WEIGHT"
echo "SCREEN_LOSS:            $SCREEN_SEGMENTATION_LOSS"
echo "SCREEN_CLASS_WEIGHT:    $SCREEN_CLASS_WEIGHT_MODE"
echo "SCREEN_BOUNDARY_WEIGHT: $SCREEN_BOUNDARY_LOSS_WEIGHT"
echo "============================================================================"

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    script_name="$(basename "$script_path")"
    echo "========================================================================"
    echo "Starting $script_name on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash "$script_path"
done
