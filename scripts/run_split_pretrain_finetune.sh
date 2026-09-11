#!/bin/bash
# 兼容入口：仅根据 KIND 分发到任务专属脚本，所有环境变量原样透传。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND="${KIND:-}"

case "${KIND}" in
    classification)
        TARGET_SCRIPT="${SCRIPT_DIR}/run_split_pretrain_finetune_classification.sh"
        ;;
    segmentation)
        TARGET_SCRIPT="${SCRIPT_DIR}/run_split_pretrain_finetune_segmentation.sh"
        ;;
    detection)
        TARGET_SCRIPT="${SCRIPT_DIR}/run_split_pretrain_finetune_detection.sh"
        ;;
    *)
        echo "KIND 必须显式设为 classification、segmentation 或 detection。" >&2
        echo "示例: KIND=classification DATA_ROOT=/path/to/data bash $0" >&2
        exit 2
        ;;
esac

echo "Dispatching KIND=${KIND} -> ${TARGET_SCRIPT}"
exec bash "${TARGET_SCRIPT}"
