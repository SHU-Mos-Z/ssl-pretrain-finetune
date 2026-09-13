#!/bin/bash
# Step 0（GPU 加速版）：离线正则化 NMF，为每张图生成 C*/E* 缓存
#
# 本脚本是 scripts/run_offline_nmf_optimized.sh 的副本，调用
# utils/preprocessing/offline_nmf_cuda.py（offline_nmf_optimized.py 的副本），
# 在保留 #1/#2 两项无损性能优化的基础上，把 MUR 迭代主循环换成 PyTorch
# 张量运算，可以跑在 GPU（CUDA）上：
#   #1 消除每次 MUR 迭代里 `_objective()`/最终 `final_recon` 对
#      `c @ e.T` 的重复大矩阵乘计算（批处理路径默认关闭该诊断量）
#   #2 批处理路径默认跳过对 C*/E* 的全量 percentile 统计
#      （单线程、随空间尺寸线性增长，且批处理逻辑从不使用）
#   #6 单图 GPU 化：整张图的迭代主循环搬到显存里执行，数据只在
#      进/出两端各搬运一次；默认用 float32（消费级显卡 fp64 吞吐低）
#
# 原有开关（与 run_offline_nmf_optimized.sh 一致）：
#   USE_SIMPLEX=1  对 C 施加 simplex 约束（ΣK=1），缓存目录自动附加 _simplex 后缀
#   LAM_E          E 的 L2 正则强度，防止坍塌端元爆炸（simplex 模式建议 0.01~0.1）
#   E_CLAMP_MAX    E 逐元素上界（物理兜底，建议与 od_max 一致=3.0；设 0 禁用）
#   E_WARN_MAX     E.max 超过此值时打印警告
#   EXCLUDE_JSON   可选分类样本人工排除清单（顶层 list，含 class_name/stem）
#   CLASS_NAME     当前 DATA_ROOT 的类别名；默认取 DATA_ROOT 的 basename
#   DRY_RUN=1      只校验并统计人工排除，不创建缓存或执行 NMF
#   PARALLEL_WORKERS / BLAS_THREADS_PER_WORKER
#                  device 为 CUDA 且只用 1 张卡时会被 Python 侧自动强制降为 1；
#                  若 GPU_ID 指定了多张卡且 DEVICE 留空，则会启用多卡并行
#                  （见下方"限定/多卡并行"说明），不再强制降为 1
#
# 新增开关（优化 #6，均可选）：
#   GPU_ID          限定本次进程能看到的物理 GPU，脚本会据此
#                   export CUDA_VISIBLE_DEVICES=$GPU_ID：
#                     - 单卡：GPU_ID=7                → 只用 7 号卡
#                     - 多卡：GPU_ID=6,7               → 只用 6/7 两块卡
#                                                       （逗号分隔，不含空格）
#                   为空时不做任何覆盖，沿用外部已设置的 CUDA_VISIBLE_DEVICES
#                   （即"本机所有 GPU 都可见"）
#   DEVICE          传给 --device 的 torch 设备字符串，如 cuda:0 / cpu；
#                   留空（或传字面 "cuda"）表示自动选择，且自动选择**只会在
#                   GPU_ID 限定的可见范围内**进行（不会碰机器上其他卡）：
#                     - 可见范围内只有 1 块卡：直接用那一块
#                     - 可见范围内有多块卡 且 PARALLEL_WORKERS>1：启用
#                       多卡轮询并行，把图像按顺序轮询分配到这几块卡上
#                       （真正的多卡并行批处理，见下方示例）
#                     - 可见范围内有多块卡 但 PARALLEL_WORKERS=1（默认）：
#                       串行处理，每张图都实时挑当前剩余显存最多的那块卡
#                   若显式传具体索引（如 DEVICE=cuda:1，索引以可见范围重新
#                   编号），则精确使用该卡，不参与自动选择或多卡轮询
#   DTYPE           float32（默认，推荐）或 float64
#
# 用法示例（把所有分解任务限定在某一张或几张卡上，不碰机器上其他卡）：
#   GPU_ID=7                              DEVICE=<留空>   → 只用 7 号卡，串行
#   GPU_ID=6,7  PARALLEL_WORKERS=2        DEVICE=<留空>   → 6/7 两卡并行批处理
#   GPU_ID=5,6,7 PARALLEL_WORKERS=6 BLAS_THREADS_PER_WORKER=4
#                                          DEVICE=<留空>   → 5/6/7 三卡并行，
#                                          每卡同时跑 2 个进程
#
# 显存不足（CUDA error: out of memory）排查：
#   - Python 侧已经会（在 GPU_ID 限定的可见范围内）自动选空闲显存最多的
#     卡，并在真的 OOM 时清一次 torch 缓存重试一次；仍失败会报明确的中文
#     提示（而不是裸的 CUDA 报错）
#   - 若仍持续 OOM，可用 `nvidia-smi` 确认各卡占用后，用 GPU_ID/DEVICE 显式
#     指定某块确定空闲的卡，或调小 K / MAX_ITER 降低单图显存占用
#
# 终端输出同时保存到 records/offline_nmf_cuda/<时间戳>/records.txt
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
# DATA_ROOT="${DATA_ROOT:-/home/zsq/processed_data/DFS3R-main/data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_20260913}"
DATA_ROOT="${DATA_ROOT:-/home/zsq/214DataA/zsq/DFS3R-main/data/GPCC_detection_patch_512x640_overlap_0x0_native_resolution_connected_component_c8_minmax_20260913}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455/B}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455/E}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455/L}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455/M}"
# DATA_ROOT="${DATA_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455/N}"
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

# ── 进程级并行（可选）；只有 1 块可见 GPU 时会被 Python 侧自动强制为 1 ──────
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
BLAS_THREADS_PER_WORKER="${BLAS_THREADS_PER_WORKER:-0}"

# ── 优化 #6：GPU 设备选择（可选） ────────────────────────────────────────────
GPU_ID="${GPU_ID:-6,7}"     # 如 7（单卡）或 6,7（多卡，逗号分隔）；为空则不覆盖
                          # 已有的 CUDA_VISIBLE_DEVICES（=本机所有 GPU 都可见）
DEVICE="${DEVICE:-cuda:1}"     # 如 cuda:0 / cpu；为空（或 cuda）表示自动选择，且只在
                          # GPU_ID 限定的可见范围内选（见文件头注释）
DTYPE="${DTYPE:-float32}" # float32（默认）或 float64

if [ -n "${GPU_ID}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

# ── 日志目录 ─────────────────────────────────────────────────────────────────
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/offline_nmf_cuda/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

echo "DATA_ROOT  = ${DATA_ROOT}"
echo "CLASS_NAME = ${CLASS_NAME}"
echo "EXCLUDE_JSON = ${EXCLUDE_JSON:-<disabled>}"
echo "DRY_RUN=${DRY_RUN}"
echo "K=${K} L1=${L1} L2=${L2} L3=${L3} MAX_ITER=${MAX_ITER}"
echo "USE_SIMPLEX=${USE_SIMPLEX}  LAM_E=${LAM_E}  E_CLAMP_MAX=${E_CLAMP_MAX}  E_WARN_MAX=${E_WARN_MAX}"
echo "PARALLEL_WORKERS=${PARALLEL_WORKERS}  BLAS_THREADS_PER_WORKER=${BLAS_THREADS_PER_WORKER}"
echo "GPU_ID=${GPU_ID:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  DEVICE=${DEVICE:-<auto>}  DTYPE=${DTYPE}"
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

DEVICE_ARGS=()
if [ -n "${DEVICE}" ]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

PYTHONUNBUFFERED=1 python -m utils.preprocessing.offline_nmf_cuda \
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
  --dtype        "${DTYPE}" \
  "${DEVICE_ARGS[@]}" \
  "${EXCLUDE_ARGS[@]}" \
  "${DRY_RUN_FLAG[@]}" \
  "${SIMPLEX_FLAG[@]}" \
  2>&1 | tee "${SAVE_DIR}/records.txt"
