#!/usr/bin/env bash
# 一次性流水线：PUAD background-aware 预处理 -> 四个 split 的 CUDA NMF
#              -> 整体 ZIP -> SCP 上传。
#
# 默认运行：
#   bash scripts/prepare_puad_bgaware_nmf_and_upload.sh
#
# 如需覆盖 GPU/NMF 运行参数（含义与 run_offline_nmf_cuda.sh 一致）：
#   GPU_ID=6,7 DEVICE=cuda:1 PARALLEL_WORKERS=1 \
#     bash scripts/prepare_puad_bgaware_nmf_and_upload.sh
#
# 密码不会写入本文件。脚本会在上传前无回显地询问一次；也可以预先设置：
#   read -rsp "SCP password: " PUAD_SCP_PASSWORD; echo
#   export PUAD_SCP_PASSWORD
#   bash scripts/prepare_puad_bgaware_nmf_and_upload.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# 本次数据与运行位置
# -----------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-zsq_accl_mine}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/data/original/LUAD_HDR}"
OUTPUT_PARENT="${OUTPUT_PARENT:-/home/zsq/processed_data/DFS3R-main/data}"
DATASET_NAME="${DATASET_NAME:-LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval}"
OUTPUT_ROOT="${OUTPUT_PARENT}/${DATASET_NAME}"
ARCHIVE_PATH="${OUTPUT_PARENT}/${DATASET_NAME}.zip"

# 目录名只概括训练 patch 数；验证 patch 数单独列出。默认值保持原始
# fg3138_bg1569 数据集行为不变。
FOREGROUND_TRAIN_PATCHES="${FOREGROUND_TRAIN_PATCHES:-3138}"
BACKGROUND_TRAIN_PATCHES="${BACKGROUND_TRAIN_PATCHES:-1569}"
FOREGROUND_VAL_PATCHES="${FOREGROUND_VAL_PATCHES:-522}"
BACKGROUND_VAL_PATCHES="${BACKGROUND_VAL_PATCHES:-261}"

PREPROCESS_SCRIPT="${PROJECT_ROOT}/data/original/luad_preprocess_official224_bgaware.py"
NMF_SCRIPT="${PROJECT_ROOT}/scripts/run_offline_nmf_cuda.sh"
NMF_SPLITS=(train val val_scenes test)

# -----------------------------------------------------------------------------
# NMF 参数：默认值与 scripts/run_offline_nmf_cuda.sh 保持一致。
# 所有值仍可由调用本脚本时的同名环境变量覆盖。
# -----------------------------------------------------------------------------
K="${K:-16}"
L1="${L1:-5e-4}"
L2="${L2:-2e-4}"
L3="${L3:-1e-2}"
MAX_ITER="${MAX_ITER:-1000}"
USE_SIMPLEX="${USE_SIMPLEX:-1}"
LAM_E="${LAM_E:-0.05}"
E_CLAMP_MAX="${E_CLAMP_MAX:-3.0}"
E_WARN_MAX="${E_WARN_MAX:-10.0}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
BLAS_THREADS_PER_WORKER="${BLAS_THREADS_PER_WORKER:-0}"
GPU_ID="${GPU_ID:-6,7}"
DEVICE="${DEVICE:-cuda:1}"
DTYPE="${DTYPE:-float32}"

# -----------------------------------------------------------------------------
# 上传目标
# -----------------------------------------------------------------------------
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.nmb2.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-27473}"
REMOTE_DIR="${REMOTE_DIR:-/autodl-fs/data}"
REMOTE_ARCHIVE="${REMOTE_DIR}/${DATASET_NAME}.zip"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

activate_conda_environment() {
  local conda_exe conda_base
  conda_exe="$(command -v conda)"
  conda_base="$(dirname "$(dirname "$conda_exe")")"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
}

ASKPASS_FILE=""
cleanup() {
  if [[ -n "${ASKPASS_FILE}" && -f "${ASKPASS_FILE}" ]]; then
    rm -f -- "${ASKPASS_FILE}"
  fi
  unset PUAD_SCP_PASSWORD SSH_ASKPASS SSH_ASKPASS_REQUIRE DISPLAY
}
trap cleanup EXIT

configure_password_authentication() {
  if [[ -z "${PUAD_SCP_PASSWORD:-}" ]]; then
    if [[ ! -t 0 ]]; then
      die "PUAD_SCP_PASSWORD is unset and stdin is not an interactive terminal"
    fi
    read -r -s -p "Password for ${REMOTE_USER}@${REMOTE_HOST}: " PUAD_SCP_PASSWORD
    printf '\n'
  fi
  [[ -n "${PUAD_SCP_PASSWORD}" ]] || die "empty SCP password"

  # OpenSSH 在无控制终端时通过这个临时程序读取环境变量中的密码。
  # 临时文件本身不含密码，并会由 EXIT trap 删除。
  ASKPASS_FILE="$(mktemp "${TMPDIR:-/tmp}/puad-scp-askpass.XXXXXX")"
  chmod 700 "${ASKPASS_FILE}"
  printf '%s\n' \
    '#!/bin/sh' \
    'printf "%s\\n" "$PUAD_SCP_PASSWORD"' >"${ASKPASS_FILE}"
  export PUAD_SCP_PASSWORD
  export SSH_ASKPASS="${ASKPASS_FILE}"
  export SSH_ASKPASS_REQUIRE=force
  export DISPLAY="${DISPLAY:-puad-upload:0}"
}

printf '%s\n' \
  '============================================================================' \
  'PUAD preprocessing, CUDA NMF, archive, and upload pipeline' \
  "Project root:   ${PROJECT_ROOT}" \
  "Source root:    ${SOURCE_ROOT}" \
  "Output root:    ${OUTPUT_ROOT}" \
  "Archive:        ${ARCHIVE_PATH}" \
  "Remote target:  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARCHIVE}" \
  "Train patches:  foreground=${FOREGROUND_TRAIN_PATCHES}, background=${BACKGROUND_TRAIN_PATCHES}" \
  "Val patches:    foreground=${FOREGROUND_VAL_PATCHES}, background=${BACKGROUND_VAL_PATCHES}" \
  "NMF splits:     ${NMF_SPLITS[*]}" \
  "NMF GPU config: GPU_ID=${GPU_ID}, DEVICE=${DEVICE}, workers=${PARALLEL_WORKERS}" \
  '============================================================================'

require_command conda
require_command zip
require_command scp
require_command ssh
require_command setsid
[[ -f "${PREPROCESS_SCRIPT}" ]] || die "missing preprocessor: ${PREPROCESS_SCRIPT}"
[[ -f "${NMF_SCRIPT}" ]] || die "missing NMF launcher: ${NMF_SCRIPT}"
[[ -d "${SOURCE_ROOT}" ]] || die "missing PUAD source directory: ${SOURCE_ROOT}"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "output already exists: ${OUTPUT_ROOT}"
[[ ! -e "${ARCHIVE_PATH}" ]] || die "archive already exists: ${ARCHIVE_PATH}"

mkdir -p -- "${OUTPUT_PARENT}"
activate_conda_environment
cd "${PROJECT_ROOT}"

printf '\n[1/4] Build background-aware PUAD dataset\n'
python "${PREPROCESS_SCRIPT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --foreground-train-patches "${FOREGROUND_TRAIN_PATCHES}" \
  --background-train-patches "${BACKGROUND_TRAIN_PATCHES}" \
  --foreground-val-patches "${FOREGROUND_VAL_PATCHES}" \
  --background-val-patches "${BACKGROUND_VAL_PATCHES}"

[[ -d "${OUTPUT_ROOT}" ]] || die "preprocessing did not create ${OUTPUT_ROOT}"
[[ ! -e "${OUTPUT_ROOT}/PREPROCESSING_INCOMPLETE" ]] || \
  die "preprocessor left PREPROCESSING_INCOMPLETE in output"

printf '\n[2/4] Run CUDA NMF for every PUAD data split\n'
for split_name in "${NMF_SPLITS[@]}"; do
  split_root="${OUTPUT_ROOT}/${split_name}"
  [[ -d "${split_root}/images" ]] || die "missing images directory: ${split_root}/images"
  printf '\n--- NMF split: %s ---\n' "${split_name}"
  DATA_ROOT="${split_root}" \
  CLASS_NAME="${split_name}" \
  K="${K}" L1="${L1}" L2="${L2}" L3="${L3}" \
  MAX_ITER="${MAX_ITER}" USE_SIMPLEX="${USE_SIMPLEX}" \
  LAM_E="${LAM_E}" E_CLAMP_MAX="${E_CLAMP_MAX}" E_WARN_MAX="${E_WARN_MAX}" \
  PARALLEL_WORKERS="${PARALLEL_WORKERS}" \
  BLAS_THREADS_PER_WORKER="${BLAS_THREADS_PER_WORKER}" \
  GPU_ID="${GPU_ID}" DEVICE="${DEVICE}" DTYPE="${DTYPE}" \
    bash "${NMF_SCRIPT}"
done

printf '\n[3/4] Archive complete dataset directory\n'
(
  cd "${OUTPUT_PARENT}"
  zip -r -1 "${DATASET_NAME}.zip" "${DATASET_NAME}"
)
[[ -s "${ARCHIVE_PATH}" ]] || die "ZIP archive was not created correctly"
printf 'Archive size: %s bytes\n' "$(stat -c '%s' "${ARCHIVE_PATH}")"

printf '\n[4/4] Upload archive with SCP and verify remote file size\n'
configure_password_authentication
SSH_OPTIONS=(
  -o BatchMode=no
  -o PreferredAuthentications=password,keyboard-interactive
  -o PubkeyAuthentication=no
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=20
)

setsid -w scp \
  "${SSH_OPTIONS[@]}" \
  -P "${REMOTE_PORT}" \
  -- "${ARCHIVE_PATH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARCHIVE}"

local_size="$(stat -c '%s' "${ARCHIVE_PATH}")"
remote_size="$(
  setsid -w ssh \
    "${SSH_OPTIONS[@]}" \
    -p "${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}" \
    "stat -c '%s' '${REMOTE_ARCHIVE}'"
)"
[[ "${remote_size}" == "${local_size}" ]] || \
  die "remote size ${remote_size} differs from local size ${local_size}"

printf '\nPipeline completed successfully.\nLocal:  %s\nRemote: %s@%s:%s\n' \
  "${ARCHIVE_PATH}" "${REMOTE_USER}" "${REMOTE_HOST}" "${REMOTE_ARCHIVE}"
