#!/bin/bash
# 已弃用：不同任务不再共用一个 Bash 入口，以免任务专属参数相互污染。
set -euo pipefail

echo "scripts/run_split_pretrain_finetune.sh 已弃用。" >&2
echo "请根据数据类型使用以下脚本之一：" >&2
echo "  classification: scripts/run_split_pretrain_finetune_classification.sh" >&2
echo "  segmentation:   scripts/run_split_pretrain_finetune_segmentation.sh" >&2
echo "  detection:      scripts/run_split_pretrain_finetune_detection.sh" >&2
exit 2
