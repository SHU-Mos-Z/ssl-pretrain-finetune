#!/bin/bash
# Evaluate a locked detection checkpoint on in-domain or OOD COCO annotations.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
CHECKPOINT="${CHECKPOINT:-records/finetune_conditioned_detection/run/ckpt_best.pth}"

# Default OOD target: GPCC test split. Override DATA_ROOT/ANNOTATION for MDC test.
DATA_ROOT="${DATA_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax}"
ANNOTATION="${ANNOTATION:-annotations/instances_train_val_test_test.json}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
WORKERS="${WORKERS:-2}"
VISUALIZATION_SAMPLES="${VISUALIZATION_SAMPLES:-12}"
AMP="${AMP:-true}"
ALLOW_INDEX_WAVELENGTHS="${ALLOW_INDEX_WAVELENGTHS:-true}"

if [ ! -f "$CHECKPOINT" ]; then echo "Missing checkpoint: $CHECKPOINT"; exit 1; fi
if [ ! -f "$DATA_ROOT/$ANNOTATION" ]; then echo "Missing annotation: $DATA_ROOT/$ANNOTATION"; exit 1; fi

EXP_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-records/evaluate_conditioned_detection/${EXP_TIME}}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
OPTIONAL_ARGS=()
if [ "$AMP" = true ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$ALLOW_INDEX_WAVELENGTHS" = true ]; then OPTIONAL_ARGS+=(--allow-index-wavelengths); fi
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    evaluate_conditioned_detection.py \
    --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --annotation "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" --batch-size "$BATCH_SIZE_PER_GPU" --workers "$WORKERS" \
    --visualization-samples "$VISUALIZATION_SAMPLES" \
    "${OPTIONAL_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR.log"
