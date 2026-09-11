#!/bin/bash
# 检测数据集：NMF 重建误差过滤 + pretrain/finetune(train/val/test) 切分。
# 仅传递检测专属参数；底层统一调用 split_pretrain_finetune.py。
set -euo pipefail
cd "$(dirname "$0")/.."

# ── 数据集 ────────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/home/zsq/processed_data/DFS3R-main/data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax}"

# annotations/{stem}.json 的 images[0] 中用于来源隔离的字段。
DETECTION_GROUP_FIELD="${DETECTION_GROUP_FIELD:-source_stem}"

# ── NMF 缓存键：须与离线 NMF 完全一致 ───────────────────────────────────────
K="${K:-16}"
L1="${L1:-5e-4}"
L2="${L2:-2e-4}"
L3="${L3:-1e-2}"
USE_SIMPLEX="${USE_SIMPLEX:-1}"
LAM_E="${LAM_E:-0.05}"
E_CLAMP_MAX="${E_CLAMP_MAX:-3.0}"

# ── 切分参数 ──────────────────────────────────────────────────────────────────
MSE_THRESHOLD="${MSE_THRESHOLD:-1.0}"
PRETRAIN_RATIO="${PRETRAIN_RATIO:-0.0}"
FINETUNE_VAL_RATIO="${FINETUNE_VAL_RATIO:-0.15}"
FINETUNE_TEST_RATIO="${FINETUNE_TEST_RATIO:-0.15}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-}"
MANIFEST_OUT="${MANIFEST_OUT:-}"
DRY_RUN="${DRY_RUN:-0}"

# 默认在项目 ./data 下自动命名；以下三个变量可分别指定确切目录。
SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-./data}"
FINETUNE_TRAIN_OUTPUT_DIR="${FINETUNE_TRAIN_OUTPUT_DIR:-}"
FINETUNE_VAL_OUTPUT_DIR="${FINETUNE_VAL_OUTPUT_DIR:-}"
FINETUNE_TEST_OUTPUT_DIR="${FINETUNE_TEST_OUTPUT_DIR:-}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_PROGRAM="${SPLIT_PROGRAM:-split_pretrain_finetune.py}"
SPLIT_RECORD_ROOT="${SPLIT_RECORD_ROOT:-./records/split_pretrain_finetune}"

EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SPLIT_RECORD_ROOT}/detection_${EXP_TIME}"
mkdir -p "${SAVE_DIR}"

echo "DATA_ROOT             = ${DATA_ROOT}"
echo "KIND                  = detection"
echo "DETECTION_GROUP_FIELD = ${DETECTION_GROUP_FIELD}"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}"
echo "MSE_THRESHOLD=${MSE_THRESHOLD}  PRETRAIN_RATIO=${PRETRAIN_RATIO}"
echo "FINETUNE_VAL_RATIO=${FINETUNE_VAL_RATIO}  FINETUNE_TEST_RATIO=${FINETUNE_TEST_RATIO}"
echo "SEED=${SEED}  MANIFEST_OUT=${MANIFEST_OUT:-<default>}  DRY_RUN=${DRY_RUN}"
echo "SPLIT_OUTPUT_ROOT         = ${SPLIT_OUTPUT_ROOT}"
echo "FINETUNE_TRAIN_OUTPUT_DIR = ${FINETUNE_TRAIN_OUTPUT_DIR:-<auto>}"
echo "FINETUNE_VAL_OUTPUT_DIR   = ${FINETUNE_VAL_OUTPUT_DIR:-<auto>}"
echo "FINETUNE_TEST_OUTPUT_DIR  = ${FINETUNE_TEST_OUTPUT_DIR:-<auto>}"
echo "PRETRAIN_OUTPUT_DIR       = ${PRETRAIN_OUTPUT_DIR:-<auto>}"
echo "记录保存至: ${SAVE_DIR}/records.txt"

SIMPLEX_ARG=(--use-simplex)
if [ "${USE_SIMPLEX}" != "1" ]; then SIMPLEX_ARG=(--no-use-simplex); fi
RUN_DATE_ARG=()
if [ -n "${RUN_DATE}" ]; then RUN_DATE_ARG=(--run-date "${RUN_DATE}"); fi
MANIFEST_ARG=()
if [ -n "${MANIFEST_OUT}" ]; then MANIFEST_ARG=(--manifest-out "${MANIFEST_OUT}"); fi
DRY_RUN_ARG=()
if [ "${DRY_RUN}" = "1" ]; then DRY_RUN_ARG=(--dry-run); fi
OUTPUT_ARGS=(--split-output-root "${SPLIT_OUTPUT_ROOT}")
if [ -n "${FINETUNE_TRAIN_OUTPUT_DIR}" ]; then
    OUTPUT_ARGS+=(--finetune-train-output-dir "${FINETUNE_TRAIN_OUTPUT_DIR}")
fi
if [ -n "${FINETUNE_VAL_OUTPUT_DIR}" ]; then
    OUTPUT_ARGS+=(--finetune-val-output-dir "${FINETUNE_VAL_OUTPUT_DIR}")
fi
if [ -n "${FINETUNE_TEST_OUTPUT_DIR}" ]; then
    OUTPUT_ARGS+=(--finetune-test-output-dir "${FINETUNE_TEST_OUTPUT_DIR}")
fi
if [ -n "${PRETRAIN_OUTPUT_DIR}" ]; then
    OUTPUT_ARGS+=(--pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}")
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SPLIT_PROGRAM}" \
    --data-root "${DATA_ROOT}" \
    --kind detection \
    "${OUTPUT_ARGS[@]}" \
    --detection-group-field "${DETECTION_GROUP_FIELD}" \
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
