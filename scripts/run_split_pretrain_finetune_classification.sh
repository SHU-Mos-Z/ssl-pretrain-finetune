#!/bin/bash
# 分类数据集：NMF 重建误差过滤 + pretrain/finetune(train/val/test) 切分。
# 仅传递分类专属参数；底层统一调用 split_pretrain_finetune.py。
set -euo pipefail
cd "$(dirname "$0")/.."

# ── 数据集 ────────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455}"

# 留空时自动发现 DATA_ROOT 下所有含 images/ 的直接子目录。
# 也可通过空格分隔的环境变量覆盖，例如：CLASS_DIRS_TEXT="B E L M N"。
CLASS_DIRS_TEXT="${CLASS_DIRS_TEXT:-}"
read -r -a CLASS_DIRS <<< "${CLASS_DIRS_TEXT}"

# 同一原始 ROI 的多个 crop 必须进入同一个 split；设为空字符串可禁用分组。
CLASSIFICATION_GROUP_REGEX="${CLASSIFICATION_GROUP_REGEX:-^(.+)__c[0-9]+$}"

# 可选人工排除 JSON，仅分类切分支持；留空表示禁用。
EXCLUDE_JSON="${EXCLUDE_JSON:-}"

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
PRETRAIN_RATIO="${PRETRAIN_RATIO:-0.0}"
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
SAVE_DIR="${SPLIT_RECORD_ROOT}/classification_${EXP_TIME}"
mkdir -p "${SAVE_DIR}"

echo "DATA_ROOT                 = ${DATA_ROOT}"
echo "KIND                      = classification"
echo "CLASS_DIRS                = ${CLASS_DIRS[*]:-<auto>}"
echo "CLASSIFICATION_GROUP_REGEX= ${CLASSIFICATION_GROUP_REGEX:-<disabled>}"
echo "EXCLUDE_JSON              = ${EXCLUDE_JSON:-<disabled>}"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}"
echo "MSE_THRESHOLD=${MSE_THRESHOLD}  PRETRAIN_RATIO=${PRETRAIN_RATIO}"
echo "FINETUNE_VAL_RATIO=${FINETUNE_VAL_RATIO}  FINETUNE_TEST_RATIO=${FINETUNE_TEST_RATIO}"
echo "SEED=${SEED}  MANIFEST_OUT=${MANIFEST_OUT:-<default>}  DRY_RUN=${DRY_RUN}"
echo "记录保存至: ${SAVE_DIR}/records.txt"

SIMPLEX_ARG=(--use-simplex)
if [ "${USE_SIMPLEX}" != "1" ]; then SIMPLEX_ARG=(--no-use-simplex); fi

CLASS_DIRS_ARG=()
if [ "${#CLASS_DIRS[@]}" -gt 0 ]; then
    CLASS_DIRS_ARG=(--class-dirs "${CLASS_DIRS[@]}")
fi
GROUP_ARG=()
if [ -n "${CLASSIFICATION_GROUP_REGEX}" ]; then
    GROUP_ARG=(--classification-group-regex "${CLASSIFICATION_GROUP_REGEX}")
fi
EXCLUDE_ARG=()
if [ -n "${EXCLUDE_JSON}" ]; then
    if [ ! -f "${EXCLUDE_JSON}" ]; then
        echo "Missing sample exclusion JSON: ${EXCLUDE_JSON}" >&2
        exit 1
    fi
    EXCLUDE_ARG=(--exclude-json "${EXCLUDE_JSON}")
fi
RUN_DATE_ARG=()
if [ -n "${RUN_DATE}" ]; then RUN_DATE_ARG=(--run-date "${RUN_DATE}"); fi
MANIFEST_ARG=()
if [ -n "${MANIFEST_OUT}" ]; then MANIFEST_ARG=(--manifest-out "${MANIFEST_OUT}"); fi
DRY_RUN_ARG=()
if [ "${DRY_RUN}" = "1" ]; then DRY_RUN_ARG=(--dry-run); fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SPLIT_PROGRAM}" \
    --data-root "${DATA_ROOT}" \
    --kind classification \
    "${CLASS_DIRS_ARG[@]}" \
    "${GROUP_ARG[@]}" \
    "${EXCLUDE_ARG[@]}" \
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
