#!/bin/bash
# 分割数据集：NMF 重建误差过滤 + pretrain/finetune(train/val/test) 切分。
# 不传递分类或检测专属参数；底层统一调用 split_pretrain_finetune.py。
set -euo pipefail
cd "$(dirname "$0")/.."

# ── 数据集 ────────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed}"

# ── NMF 缓存键：须与离线 NMF 完全一致 ───────────────────────────────────────
K="${K:-16}"
L1="${L1:-5e-4}"
L2="${L2:-2e-4}"
L3="${L3:-1e-2}"
USE_SIMPLEX="${USE_SIMPLEX:-1}"
LAM_E="${LAM_E:-0.05}"
E_CLAMP_MAX="${E_CLAMP_MAX:-3.0}"

# ── 切分参数 ──────────────────────────────────────────────────────────────────
MSE_THRESHOLD="${MSE_THRESHOLD:-0.01}"
PRETRAIN_RATIO="${PRETRAIN_RATIO:-0.30}"
FINETUNE_VAL_RATIO="${FINETUNE_VAL_RATIO:-0.15}"
FINETUNE_TEST_RATIO="${FINETUNE_TEST_RATIO:-0.15}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-}"
MANIFEST_OUT="${MANIFEST_OUT:-}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_PROGRAM="${SPLIT_PROGRAM:-split_pretrain_finetune.py}"
SPLIT_RECORD_ROOT="${SPLIT_RECORD_ROOT:-./records/split_pretrain_finetune}"

EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SPLIT_RECORD_ROOT}/segmentation_${EXP_TIME}"
mkdir -p "${SAVE_DIR}"

echo "DATA_ROOT            = ${DATA_ROOT}"
echo "KIND                 = segmentation"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}"
echo "MSE_THRESHOLD=${MSE_THRESHOLD}  PRETRAIN_RATIO=${PRETRAIN_RATIO}"
echo "FINETUNE_VAL_RATIO=${FINETUNE_VAL_RATIO}  FINETUNE_TEST_RATIO=${FINETUNE_TEST_RATIO}"
echo "SEED=${SEED}  MANIFEST_OUT=${MANIFEST_OUT:-<default>}  DRY_RUN=${DRY_RUN}"
echo "记录保存至: ${SAVE_DIR}/records.txt"

SIMPLEX_ARG=(--use-simplex)
if [ "${USE_SIMPLEX}" != "1" ]; then SIMPLEX_ARG=(--no-use-simplex); fi
RUN_DATE_ARG=()
if [ -n "${RUN_DATE}" ]; then RUN_DATE_ARG=(--run-date "${RUN_DATE}"); fi
MANIFEST_ARG=()
if [ -n "${MANIFEST_OUT}" ]; then MANIFEST_ARG=(--manifest-out "${MANIFEST_OUT}"); fi
DRY_RUN_ARG=()
if [ "${DRY_RUN}" = "1" ]; then DRY_RUN_ARG=(--dry-run); fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SPLIT_PROGRAM}" \
    --data-root "${DATA_ROOT}" \
    --kind segmentation \
    --k "${K}" --l1 "${L1}" --l2 "${L2}" --l3 "${L3}" \
    "${SIMPLEX_ARG[@]}" --lam-e "${LAM_E}" --e-clamp-max "${E_CLAMP_MAX}" \
    --mse-threshold "${MSE_THRESHOLD}" \
    --pretrain-ratio "${PRETRAIN_RATIO}" \
    --finetune-val-ratio "${FINETUNE_VAL_RATIO}" \
    --finetune-test-ratio "${FINETUNE_TEST_RATIO}" \
    --seed "${SEED}" \
    "${RUN_DATE_ARG[@]}" \
    "${MANIFEST_ARG[@]}" \
    "${DRY_RUN_ARG[@]}" \
    2>&1 | tee "${SAVE_DIR}/records.txt"
