#!/bin/bash
# Optional PUAD experiment: compute one E* per original scene.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
DATASET_ROOT="${DATASET_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATASET_ROOT/scene_endmembers_K16_l10.0005_l20.0002_l30.01_le0.05_ec3_simplex}"
NMF_K="${NMF_K:-16}"
NMF_L1="${NMF_L1:-5e-4}"
NMF_L2="${NMF_L2:-2e-4}"
NMF_L3="${NMF_L3:-1e-2}"
NMF_LAM_E="${NMF_LAM_E:-0.05}"
NMF_E_CLAMP_MAX="${NMF_E_CLAMP_MAX:-3.0}"
OD_MAX="${OD_MAX:-3.0}"
MAX_ITER="${MAX_ITER:-500}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
MAX_SCENES="${MAX_SCENES:-0}"

python scripts/preprocessing/puad_prepare_scene_endmembers.py \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --k "$NMF_K" \
    --l1 "$NMF_L1" \
    --l2 "$NMF_L2" \
    --l3 "$NMF_L3" \
    --lam-e "$NMF_LAM_E" \
    --e-clamp-max "$NMF_E_CLAMP_MAX" \
    --od-max "$OD_MAX" \
    --max-iter "$MAX_ITER" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-scenes "$MAX_SCENES"
