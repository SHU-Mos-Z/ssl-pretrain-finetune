#!/bin/bash
# GPCC 0913 E04: retinanet + gated_pyramid + legacy head.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "scripts/experiment_naming.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,7}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_ROOT="${TRAIN_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_finetune_train_p072_20260913}"
VAL_ROOT="${VAL_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_finetune_val_p014_20260913}"
TEST_ROOT="${TEST_ROOT:-data/GPCC_detection_patch_512x640_overlap_0x0_to_256x320_connected_component_c8_minmax_finetune_test_p014_20260913}"
TRAIN_ANNOTATION="${TRAIN_ANNOTATION:-annotations}"
VAL_ANNOTATION="${VAL_ANNOTATION:-annotations}"
TEST_ANNOTATION="${TEST_ANNOTATION:-annotations}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"

DETECTION_MODE="anchor_based"
DET_FEATURE_MODE="gated_pyramid"
DET_HEAD_NORM="none"
DET_HEAD_NORM_GROUPS="${DET_HEAD_NORM_GROUPS:-32}"
DET_QUALITY_MODE="legacy"
QUALITY_LOSS_WEIGHT="${QUALITY_LOSS_WEIGHT:-1.0}"
QUALITY_SCORE_POWER="${QUALITY_SCORE_POWER:-0.5}"
DETECTION_VIEW_MODE="direct"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-4e-4}"
BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.01}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-42}"
AMP="${AMP:-true}"
AUGMENT="${AUGMENT:-true}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-false}"

PATCH_SIZE="${PATCH_SIZE:-16}"
SPECTRAL_PATCH_SIZE="${SPECTRAL_PATCH_SIZE:-5}"
EMBED_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
MLP_RATIO=4.0
DROPOUT=0.1
CNN_STEM_CH=64
CNN_SPECTRAL_AGG="attention"
FUSION_HEADS=8
FEATURE_DIM=128
DECODER_MID_CH=64
RESIDUAL_HIDDEN_DIM=128
RIDGE_LAMBDA=1e-3
CONFIDENCE_TEMPERATURE=0.05
ALPHA_MIN=0.1
ALPHA_EXTRA=1.0
OD_MAX=3.0

DET_FEATURE_DIM="${DET_FEATURE_DIM:-128}"
HEAD_DEPTH="${HEAD_DEPTH:-4}"
HEAD_NORM_GROUPS="${HEAD_NORM_GROUPS:-$DET_HEAD_NORM_GROUPS}"
POSITIVE_IOU=0.5
NEGATIVE_IOU=0.4
IGNORE_IOU=0.5
BOX_LOSS="giou"
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
CENTERNESS_LOSS_WEIGHT=1.0

AP_SCORE_THRESHOLD="${AP_SCORE_THRESHOLD:-0.05}"
DEPLOY_SCORE_THRESHOLD="${DEPLOY_SCORE_THRESHOLD:-}"
VIS_SCORE_THRESHOLD="${VIS_SCORE_THRESHOLD:-}"
VIS_MAX_DETECTIONS="${VIS_MAX_DETECTIONS:-30}"
THRESHOLD_CALIBRATION_IOU="${THRESHOLD_CALIBRATION_IOU:-0.5}"
THRESHOLD_SEARCH_MIN="${THRESHOLD_SEARCH_MIN:-0.05}"
THRESHOLD_SEARCH_MAX="${THRESHOLD_SEARCH_MAX:-0.90}"
THRESHOLD_SEARCH_STEP="${THRESHOLD_SEARCH_STEP:-0.01}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}"
PRE_NMS_TOPK="${PRE_NMS_TOPK:-1000}"
MAX_DETECTIONS="${MAX_DETECTIONS:-100}"

NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
NMF_CACHE_NAME="nmf_cache_K16_l15e-4_l22e-4_l31e-2_le0.05_ec3_simplex"
ALLOW_INDEX_WAVELENGTHS=true
WAVELENGTH_FILE=""

WORKERS="${WORKERS:-2}"
SAVE_INTERVAL=10
EVALUATION_INTERVAL=5
PR_CURVE_INTERVAL=5
TEST_VISUALIZATION_SAMPLES="${TEST_VISUALIZATION_SAMPLES:-12}"
PROGRESS="${PROGRESS:-log}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    for required in images masks ignore_masks annotations "$NMF_CACHE_NAME"; do
        if [ ! -d "$root/$required" ]; then
            echo "Missing required GPCC detection input: $root/$required" >&2
            exit 1
        fi
    done
    if [ ! -f "$root/wavelengths.npy" ]; then
        echo "Missing wavelength metadata: $root/wavelengths.npy" >&2
        exit 1
    fi
done
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing pretraining checkpoint: $PRETRAIN_CKPT" >&2
    exit 1
fi

DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")"
INPUT_SHAPE="$("$PYTHON_BIN" -c 'from pathlib import Path; import numpy as np, sys; p=next((Path(sys.argv[1])/"images").glob("*.npy")); a=np.load(p, mmap_mode="r"); print(a.shape[0], a.shape[1])' "$TRAIN_ROOT")"
read -r INPUT_HEIGHT INPUT_WIDTH <<< "$INPUT_SHAPE"
INPUT_SIZE="${INPUT_HEIGHT}x${INPUT_WIDTH}"
if [[ "$DATASET_INFO" != *"$INPUT_SIZE"* ]]; then
    DATASET_INFO="${DATASET_INFO}-${INPUT_SIZE}"
fi
if (( INPUT_HEIGHT % PATCH_SIZE != 0 || INPUT_WIDTH % PATCH_SIZE != 0 )); then
    echo "Input size $INPUT_SIZE is not divisible by PATCH_SIZE=$PATCH_SIZE" >&2
    exit 1
fi

echo "TRAIN_ROOT=$TRAIN_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
echo "INPUT_SIZE=$INPUT_SIZE DETECTION_MODE=$DETECTION_MODE FEATURE=$DET_FEATURE_MODE HEAD_NORM=$DET_HEAD_NORM QUALITY=$DET_QUALITY_MODE"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: inputs and configuration passed validation; training was not started."
    exit 0
fi

ANCHOR_CONFIG_JSON="${ANCHOR_CONFIG_JSON:-records/test_detection_plgc_0913/anchor_fit_${DATASET_INFO}_seed${SEED}.json}"
mkdir -p "$(dirname "$ANCHOR_CONFIG_JSON")"
if [ ! -f "$ANCHOR_CONFIG_JSON" ]; then
    "$PYTHON_BIN" scripts/fit_wbc_detection_anchors.py \
        --data-root "$TRAIN_ROOT" --annotation "$TRAIN_ANNOTATION" \
        --source-crop-size "$INPUT_HEIGHT,$INPUT_WIDTH" \
        --model-input-size "$INPUT_HEIGHT,$INPUT_WIDTH" \
        --views-per-source 1 --simulation-epochs 1 \
        --positive-guided-fraction 1.0 --visible-ratio 0.0 --min-visible-side 0.0 \
        --seed "$SEED" --output "$ANCHOR_CONFIG_JSON"
fi
DETECTOR_GEOMETRY_ARGS=(--anchor-config-json "$ANCHOR_CONFIG_JSON")


EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="records/test_detection_plgc_0913/GPCC-E04_${DATASET_INFO}-direct-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}_retinanet-gated_pyramid-legacy_seed${SEED}_${EXP_TIME}"
mkdir -p "$SAVE_DIR"
echo "SAVE_DIR=$SAVE_DIR"

OPTIONAL_ARGS=()
if [ "$AMP" = true ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$AUGMENT" = true ]; then OPTIONAL_ARGS+=(--augment); else OPTIONAL_ARGS+=(--no-augment); fi
if [ "$FREEZE_BACKBONE" = true ]; then OPTIONAL_ARGS+=(--freeze-backbone); fi
if [ "$NMF_SIMPLEX" = true ]; then OPTIONAL_ARGS+=(--nmf-simplex); else OPTIONAL_ARGS+=(--no-nmf-simplex); fi
if [ "$ALLOW_INDEX_WAVELENGTHS" = true ]; then OPTIONAL_ARGS+=(--allow-index-wavelengths); fi
if [ -n "$WAVELENGTH_FILE" ]; then OPTIONAL_ARGS+=(--wavelength-file "$WAVELENGTH_FILE"); fi
if [ -n "$DEPLOY_SCORE_THRESHOLD" ]; then OPTIONAL_ARGS+=(--deployment-score-threshold "$DEPLOY_SCORE_THRESHOLD"); fi
if [ -n "$VIS_SCORE_THRESHOLD" ]; then OPTIONAL_ARGS+=(--visualization-score-threshold "$VIS_SCORE_THRESHOLD"); fi

MASTER_PORT=$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=2 torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train_finetune_conditioned_detection.py \
    --train-root "$TRAIN_ROOT" --train-annotation "$TRAIN_ANNOTATION" \
    --val-root "$VAL_ROOT" --val-annotation "$VAL_ANNOTATION" \
    --test-root "$TEST_ROOT" --test-annotation "$TEST_ANNOTATION" \
    --pretrain-ckpt "$PRETRAIN_CKPT" \
    --detection-mode "$DETECTION_MODE" --det-feature-mode "$DET_FEATURE_MODE" \
    --det-head-norm "$DET_HEAD_NORM" --det-head-norm-groups "$HEAD_NORM_GROUPS" \
    --det-quality-mode "$DET_QUALITY_MODE" --quality-loss-weight "$QUALITY_LOSS_WEIGHT" \
    --quality-score-power "$QUALITY_SCORE_POWER" --detection-view-mode "$DETECTION_VIEW_MODE" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE_PER_GPU" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --backbone-lr-mult "$BACKBONE_LR_MULT" --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS" \
    --clip-grad "$CLIP_GRAD" --workers "$WORKERS" --seed "$SEED" \
    --save-interval "$SAVE_INTERVAL" --evaluation-interval "$EVALUATION_INTERVAL" \
    --pr-curve-interval "$PR_CURVE_INTERVAL" \
    --test-visualization-samples "$TEST_VISUALIZATION_SAMPLES" \
    --visualization-max-detections "$VIS_MAX_DETECTIONS" \
    --threshold-calibration-iou "$THRESHOLD_CALIBRATION_IOU" \
    --threshold-search-min "$THRESHOLD_SEARCH_MIN" --threshold-search-max "$THRESHOLD_SEARCH_MAX" \
    --threshold-search-step "$THRESHOLD_SEARCH_STEP" \
    --progress "$PROGRESS" --log-interval "$LOG_INTERVAL" \
    --patch-size "$PATCH_SIZE" --spectral-patch-size "$SPECTRAL_PATCH_SIZE" \
    --embed-dim "$EMBED_DIM" --vit-depth "$VIT_DEPTH" --vit-heads "$VIT_HEADS" \
    --mlp-ratio "$MLP_RATIO" --dropout "$DROPOUT" --cnn-stem-ch "$CNN_STEM_CH" \
    --cnn-spectral-agg "$CNN_SPECTRAL_AGG" --fusion-heads "$FUSION_HEADS" \
    --feature-dim "$FEATURE_DIM" --decoder-mid-ch "$DECODER_MID_CH" \
    --residual-hidden-dim "$RESIDUAL_HIDDEN_DIM" --ridge-lambda "$RIDGE_LAMBDA" \
    --confidence-temperature "$CONFIDENCE_TEMPERATURE" --alpha-min "$ALPHA_MIN" \
    --alpha-extra "$ALPHA_EXTRA" --od-max "$OD_MAX" \
    --det-feature-dim "$DET_FEATURE_DIM" --head-depth "$HEAD_DEPTH" \
    "${DETECTOR_GEOMETRY_ARGS[@]}" \
    --positive-iou-threshold "$POSITIVE_IOU" --negative-iou-threshold "$NEGATIVE_IOU" \
    --ignore-iou-threshold "$IGNORE_IOU" --box-loss "$BOX_LOSS" \
    --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" \
    --centerness-loss-weight "$CENTERNESS_LOSS_WEIGHT" \
    --ap-score-threshold "$AP_SCORE_THRESHOLD" --nms-threshold "$NMS_THRESHOLD" \
    --pre-nms-topk "$PRE_NMS_TOPK" --max-detections "$MAX_DETECTIONS" \
    --nmf-k "$NMF_K" --nmf-l1 "$NMF_L1" --nmf-l2 "$NMF_L2" --nmf-l3 "$NMF_L3" \
    --nmf-lam-e "$NMF_LAM_E" --nmf-e-clamp-max "$NMF_E_CLAMP_MAX" \
    --save-dir "$SAVE_DIR" "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"
