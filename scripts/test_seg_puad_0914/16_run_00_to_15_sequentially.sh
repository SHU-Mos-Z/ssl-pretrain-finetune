#!/bin/bash
# Run the PUAD segmentation experiment matrix sequentially on one fixed GPU set.
set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-10}"
# Data-pipeline-only controls.  They do not change sample order, augmentation
# parameters, model inputs, loss, optimizer, or learning-rate schedule.
export WORKERS="${WORKERS:-2}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export DISTRIBUTED_VALIDATION="${DISTRIBUTED_VALIDATION:-true}"
# 统一传给全部单项实验。当前默认 1 GPU × 8/GPU，总 batch=8，沿用
# 参考设置 2 GPU × 4/GPU、LR=4e-4 的相同全局 batch 与学习率。
# 可在执行前覆盖，例如：LR=4e-4 bash 16_run_00_to_15_sequentially.sh
export LR="${LR:-5e-4}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
FAST_PUAD_ROOT="${FAST_PUAD_ROOT:-/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval}"
export PUAD_DATA_ROOT="${PUAD_DATA_ROOT:-${FAST_PUAD_ROOT}}"
export TRAIN_ROOT="${TRAIN_ROOT:-${FAST_PUAD_ROOT}/train}"
export VAL_ROOT="${VAL_ROOT:-${FAST_PUAD_ROOT}/val}"
export SCENE_VAL_ROOT="${SCENE_VAL_ROOT:-${FAST_PUAD_ROOT}/val_scenes}"
export TEST_ROOT="${TEST_ROOT:-${FAST_PUAD_ROOT}/test}"
export SCENE_VAL_INTERVAL="${SCENE_VAL_INTERVAL:-10}"
export SCENE_VAL_WINDOW_SIZE="${SCENE_VAL_WINDOW_SIZE:-224}"
export SCENE_VAL_WINDOW_STRIDE="${SCENE_VAL_WINDOW_STRIDE:-112}"
export SCENE_VAL_WINDOW_BATCH_SIZE="${SCENE_VAL_WINDOW_BATCH_SIZE:-4}"
export SCENE_VAL_WINDOW_BLEND="${SCENE_VAL_WINDOW_BLEND:-gaussian}"

# 新数据协议首次运行：依次执行完整的 00--15 配置矩阵。
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
echo "WORKERS_PER_RANK:       $WORKERS"
echo "PERSISTENT_WORKERS:     $PERSISTENT_WORKERS"
echo "PREFETCH_FACTOR:        $PREFETCH_FACTOR"
echo "DISTRIBUTED_VALIDATION: $DISTRIBUTED_VALIDATION"
echo "LR:                     $LR"
echo "PRETRAIN_CKPT:          $PRETRAIN_CKPT"
echo "TRAIN_ROOT:             $TRAIN_ROOT"
echo "VAL_ROOT:               $VAL_ROOT"
echo "SCENE_VAL_ROOT:         $SCENE_VAL_ROOT"
echo "SCENE_VAL_INTERVAL:     $SCENE_VAL_INTERVAL"
echo "TEST_ROOT:              $TEST_ROOT"
echo "============================================================================"

SCENE_CACHE="${SCENE_ENDMEMBER_ROOT:-${FAST_PUAD_ROOT}/scene_endmembers_K16_l10.0005_l20.0002_l30.01_le0.05_ec3_simplex}"

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
