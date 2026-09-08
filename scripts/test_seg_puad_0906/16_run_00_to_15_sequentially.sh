#!/bin/bash
# Run the PUAD segmentation experiment matrix sequentially on one fixed GPU set.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
# 统一传给全部单项实验。参考设置为 2 GPU × 4/GPU、LR=4e-4；当前默认
# 2 GPU × 8/GPU 的总 batch 翻倍，因此按线性缩放默认使用 8e-4。
# 可在执行前覆盖，例如：LR=4e-4 bash 16_run_00_to_15_sequentially.sh
export LR="${LR:-8e-4}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660/train}"
export VAL_ROOT="${VAL_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660/val}"
export TEST_ROOT="${TEST_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660/test}"

EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/00_p_00_default.sh"
    "$SCRIPT_DIR/01_p_a1_dihedral.sh"
    "$SCRIPT_DIR/02_p_a2_dihedral_affine.sh"
    "$SCRIPT_DIR/03_p_a3_dihedral_perspective.sh"
    "$SCRIPT_DIR/04_p_h1_residual.sh"
    "$SCRIPT_DIR/05_p_h2_aspp.sh"
    "$SCRIPT_DIR/06_p_h3_multiscale_aux.sh"
    "$SCRIPT_DIR/07_p_l1_weighted_ce_dice.sh"
    "$SCRIPT_DIR/08_p_l2_focal_dice.sh"
    "$SCRIPT_DIR/09_p_l3_weighted_boundary.sh"
    "$SCRIPT_DIR/10_p_o1_discriminative_lr.sh"
    "$SCRIPT_DIR/11_p_o2_staged_unfreeze.sh"
    "$SCRIPT_DIR/12_p_e1_scene_endmembers.sh"
    "$SCRIPT_DIR/13_p_final_seed42.sh"
    "$SCRIPT_DIR/14_p_final_seed43.sh"
    "$SCRIPT_DIR/15_p_final_seed44.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    [ -f "$script_path" ] || { echo "Missing experiment script: $script_path"; exit 2; }
done

echo "============================================================================"
echo "PUAD sequential experiment runner"
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
echo "============================================================================"

SCENE_CACHE="${SCENE_ENDMEMBER_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660/scene_endmembers_K16_l10.0005_l20.0002_l30.01_le0.05_ec3_simplex}"

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    script_name=$(basename "$script_path")
    if [ "$script_name" = "12_p_e1_scene_endmembers.sh" ]; then
        if [ -d "$SCENE_CACHE" ] && compgen -G "$SCENE_CACHE/*_E.npy" >/dev/null; then
            export SCENE_ENDMEMBER_ROOT="$SCENE_CACHE"
        else
            echo "Skipping P-E1: scene-level E* cache is not ready at $SCENE_CACHE"
            continue
        fi
    fi
    echo "========================================================================"
    echo "Starting $script_name on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash "$script_path"
done
