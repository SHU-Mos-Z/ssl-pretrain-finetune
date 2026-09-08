#!/bin/bash
# Step 0：离线正则化 NMF，为每张图生成 C*/E* 缓存
#
# 新增开关：
#   USE_SIMPLEX=1  对 C 施加 simplex 约束（ΣK=1），缓存目录自动附加 _simplex 后缀
#   LAM_E          E 的 L2 正则强度，防止坍塌端元爆炸（simplex 模式建议 0.01~0.1）
#   E_CLAMP_MAX    E 逐元素上界（物理兜底，建议与 od_max 一致=3.0；设 0 禁用）
#   E_WARN_MAX     E.max 超过此值时打印警告
#   EXCLUDE_JSON   可选分类样本人工排除清单（顶层 list，含 class_name/stem）
#   CLASS_NAME     当前 DATA_ROOT 的类别名；默认取 DATA_ROOT 的 basename
#   DRY_RUN=1      只校验并统计人工排除，不创建缓存或执行 NMF
#
# 终端输出同时保存到 records/offline_nmf/<时间戳>/records.txt
# 批量完成后，NMF 缓存目录会自动写入 nmf_reconstruction_mse.json（每张图的重建 MSE 索引）

set -euo pipefail
cd "$(dirname "$0")/.."

# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50/N}"
# DATA_ROOT="${DATA_ROOT:-data/PLGC_class_patch_512x512_overlap_0x0_to_256x256_minmax/Normal}"
# DATA_ROOT="${DATA_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x256_xml_object_minmax}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_1300x1800_overlap_0x0_to_256x256_minmax_bands50/B}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_1300x1800_overlap_0x0_to_256x256_minmax_bands50/E}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_1300x1800_overlap_0x0_to_256x256_minmax_bands50/L}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_1300x1800_overlap_0x0_to_256x256_minmax_bands50/M}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_patch_1300x1800_overlap_0x0_to_256x256_minmax_bands50/N}"
# DATA_ROOT="${DATA_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x320_xml_object_minmax}"
# DATA_ROOT="${DATA_ROOT:-data/MDC_detection_patch_1024x1280_overlap_0x0_to_256x256_xml_object_minmax_maskflipped}"
# DATA_ROOT="${data/LUAD_PUAD_official224_centerbalanced_3660/val}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830/B}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830/E}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830/L}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830/M}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830/N}"
DATA_ROOT="${DATA_ROOT:-data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725}"
# DATA_ROOT="${DATA_ROOT:-data/LUAD_PUAD_official224_centerbalanced_3660/train}"
# DATA_ROOT="${DATA_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax}"
# DATA_ROOT="${DATA_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x256_connected_component_c8_minmax}"
# DATA_ROOT="${DATA_ROOT:-data/TMA_patch_1024x1024_overlap_0x0_to_256x256_minmax}"
# DATA_ROOT="${DATA_ROOT:-data/LUAD_patch_400x400_overlap_0x0_to_256x256_minmax/Training}"
# DATA_ROOT="${DATA_ROOT:-data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed}"
# DATA_ROOT="${DATA_ROOT:-data/PDAC_pretrain_preprocessed}"
# DATA_ROOT="${DATA_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed}"
K="${K:-16}"
L1="${L1:-5e-4}"
L2="${L2:-2e-4}"
L3="${L3:-1e-2}"
MAX_ITER="${MAX_ITER:-1000}"
USE_SIMPLEX="${USE_SIMPLEX:-1}"   # 1=开启 simplex 约束（推荐），0=关闭
LAM_E="${LAM_E:-0.05}"           # E L2 正则（simplex 模式防端元爆炸）
E_CLAMP_MAX="${E_CLAMP_MAX:-3.0}" # E 上界（<=0 表示禁用）
E_WARN_MAX="${E_WARN_MAX:-10.0}"  # E.max 警告阈值
EXCLUDE_JSON="${EXCLUDE_JSON:-}"
CLASS_NAME="${CLASS_NAME:-$(basename "$DATA_ROOT")}"
DRY_RUN="${DRY_RUN:-0}"

# ── 日志目录 ─────────────────────────────────────────────────────────────────
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/offline_nmf/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

echo "DATA_ROOT  = ${DATA_ROOT}"
echo "CLASS_NAME = ${CLASS_NAME}"
echo "EXCLUDE_JSON = ${EXCLUDE_JSON:-<disabled>}"
echo "DRY_RUN=${DRY_RUN}"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3} MAX_ITER=${MAX_ITER}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}  E_WARN_MAX=${E_WARN_MAX}"
echo "记录保存至: ${SAVE_DIR}/records.txt"

SIMPLEX_FLAG=()
if [ "${USE_SIMPLEX}" = "1" ]; then
  SIMPLEX_FLAG=(--use-simplex)
fi

EXCLUDE_ARGS=()
if [ -n "${EXCLUDE_JSON}" ]; then
  if [ ! -f "${EXCLUDE_JSON}" ]; then
    echo "Missing sample exclusion JSON: ${EXCLUDE_JSON}"
    exit 1
  fi
  EXCLUDE_ARGS=(--exclude-json "${EXCLUDE_JSON}" --class-name "${CLASS_NAME}")
fi

DRY_RUN_FLAG=()
if [ "${DRY_RUN}" = "1" ]; then
  DRY_RUN_FLAG=(--dry-run)
elif [ "${DRY_RUN}" != "0" ]; then
  echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}"
  exit 1
fi

PYTHONUNBUFFERED=1 python -m utils.preprocessing.offline_nmf \
  --data-root    "${DATA_ROOT}" \
  --k            "${K}" \
  --l1           "${L1}" \
  --l2           "${L2}" \
  --l3           "${L3}" \
  --max-iter     "${MAX_ITER}" \
  --lam-e        "${LAM_E}" \
  --e-clamp-max  "${E_CLAMP_MAX}" \
  --e-warn-max   "${E_WARN_MAX}" \
  "${EXCLUDE_ARGS[@]}" \
  "${DRY_RUN_FLAG[@]}" \
  "${SIMPLEX_FLAG[@]}" \
  2>&1 | tee "${SAVE_DIR}/records.txt"
