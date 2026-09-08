#!/bin/bash
# Step 0（性能优化版）：离线正则化 NMF，为每张图生成 C*/E* 缓存
#
# 本脚本是 scripts/run_offline_nmf.sh 的副本，调用
# utils/preprocessing/offline_nmf_optimized.py（offline_nmf.py 的副本），
# 在数值完全等价的前提下落实了三项性能优化：
#   #1 消除每次 MUR 迭代里 `_objective()`/最终 `final_recon` 对
#      `c @ e.T` 的重复大矩阵乘计算（批处理路径默认关闭该诊断量）
#   #2 批处理路径默认跳过对 C*/E* 的全量 percentile 统计
#      （单线程、随空间尺寸线性增长，且批处理逻辑从不使用）
#   #5 新增进程级并行开关：图与图之间天然独立，可选择用多进程
#      同时处理多张图，更好地利用多核 CPU
#
# 原有开关（与 run_offline_nmf.sh 一致）：
#   USE_SIMPLEX=1  对 C 施加 simplex 约束（ΣK=1），缓存目录自动附加 _simplex 后缀
#   LAM_E          E 的 L2 正则强度，防止坍塌端元爆炸（simplex 模式建议 0.01~0.1）
#   E_CLAMP_MAX    E 逐元素上界（物理兜底，建议与 od_max 一致=3.0；设 0 禁用）
#   E_WARN_MAX     E.max 超过此值时打印警告
#   EXCLUDE_JSON   可选分类样本人工排除清单（顶层 list，含 class_name/stem）
#   CLASS_NAME     当前 DATA_ROOT 的类别名；默认取 DATA_ROOT 的 basename
#   DRY_RUN=1      只校验并统计人工排除，不创建缓存或执行 NMF
#
# 新增开关（优化 #5，均可选，默认关闭=等价于原始串行行为）：
#   PARALLEL_WORKERS=N        大于 1 时启用进程级并行，同时处理 N 张图
#   BLAS_THREADS_PER_WORKER=M 配合并行使用：限制每个 worker 进程的 BLAS
#                              线程数（建议 4~8），避免与进程级并行抢核；
#                              留空/0 表示不限制
#
# 终端输出同时保存到 records/offline_nmf_optimized/<时间戳>/records.txt
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

# ── 优化 #5：进程级并行（可选） ──────────────────────────────────────────────
PARALLEL_WORKERS="${PARALLEL_WORKERS:-2}"              # 1=不启用并行（默认，等价原始串行）
BLAS_THREADS_PER_WORKER="${BLAS_THREADS_PER_WORKER:-0}" # 0=不限制每个 worker 的 BLAS 线程数

# ── 日志目录 ─────────────────────────────────────────────────────────────────
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/offline_nmf_optimized/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

echo "DATA_ROOT  = ${DATA_ROOT}"
echo "CLASS_NAME = ${CLASS_NAME}"
echo "EXCLUDE_JSON = ${EXCLUDE_JSON:-<disabled>}"
echo "DRY_RUN=${DRY_RUN}"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3} MAX_ITER=${MAX_ITER}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}  E_WARN_MAX=${E_WARN_MAX}"
echo "PARALLEL_WORKERS=${PARALLEL_WORKERS}  BLAS_THREADS_PER_WORKER=${BLAS_THREADS_PER_WORKER}"
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

PYTHONUNBUFFERED=1 python -m utils.preprocessing.offline_nmf_optimized \
  --data-root    "${DATA_ROOT}" \
  --k            "${K}" \
  --l1           "${L1}" \
  --l2           "${L2}" \
  --l3           "${L3}" \
  --max-iter     "${MAX_ITER}" \
  --lam-e        "${LAM_E}" \
  --e-clamp-max  "${E_CLAMP_MAX}" \
  --e-warn-max   "${E_WARN_MAX}" \
  --parallel-workers          "${PARALLEL_WORKERS}" \
  --blas-threads-per-worker   "${BLAS_THREADS_PER_WORKER}" \
  "${EXCLUDE_ARGS[@]}" \
  "${DRY_RUN_FLAG[@]}" \
  "${SIMPLEX_FLAG[@]}" \
  2>&1 | tee "${SAVE_DIR}/records.txt"
