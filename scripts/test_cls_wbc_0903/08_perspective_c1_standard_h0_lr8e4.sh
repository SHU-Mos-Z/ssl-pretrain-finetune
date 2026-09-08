#!/bin/bash
# W3-08: perspective copies=1, standard sampling, H0 on 256x256x50 focused-cell WBC data.
# Complete standalone two-GPU classification training script.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"
source "scripts/experiment_naming.sh"

# ── Experiment-specific variables ─────────────────────────────────────────────
EXPERIMENT_ID="W3-08"
AUGMENT="true"
AUGMENTATION_COPIES="1"
AUGMENTATION_POLICY="perspective"
PERSPECTIVE_PROBABILITY="1.0"
PERSPECTIVE_SCALE="0.05"
PERSPECTIVE_PADDING_MODE="reflection"
BALANCE_MODE="none"  # none | balanced_sampling | inverse_freq_loss
CLASSIFICATION_HEAD="h0_gap_linear"
DEFAULT_LR="8e-4"
SEED="42"

# ── Hardware ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# ── Manually selected dataset splits ──────────────────────────────────────────
# Fixed WBC-0903 finetune-only p070/p015/p015 split; this script never scans data/ for a newer split.
TRAIN_ROOT="${TRAIN_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_train_p070_20260903}"
VAL_ROOT="${VAL_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_val_p015_20260903}"
TEST_ROOT="${TEST_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_test_p015_20260903}"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ ! -d "$root" ]; then
        echo "Missing WBC classification split: $root"
        exit 1
    fi
done

# ── Sampling and loss balance ─────────────────────────────────────────────────
case "$BALANCE_MODE" in
    none)
        CLASS_WEIGHTING="none"
        SAMPLING_STRATEGY="standard"
        ;;
    balanced_sampling)
        CLASS_WEIGHTING="none"
        SAMPLING_STRATEGY="balanced"
        ;;
    inverse_freq_loss)
        CLASS_WEIGHTING="inverse_freq"
        SAMPLING_STRATEGY="standard"
        ;;
    *)
        echo "BALANCE_MODE must be none, balanced_sampling, or inverse_freq_loss."
        exit 1
        ;;
esac

case "$AUGMENTATION_POLICY" in
    dihedral|perspective|dihedral_perspective) ;;
    *)
        echo "Invalid AUGMENTATION_POLICY: $AUGMENTATION_POLICY"
        exit 1
        ;;
esac
if [ "$AUGMENT" != "true" ] && [ "$AUGMENTATION_COPIES" != "1" ]; then
    echo "AUGMENTATION_COPIES must equal 1 when AUGMENT=false."
    exit 1
fi
if ! [[ "$AUGMENTATION_COPIES" =~ ^[1-8]$ ]]; then
    echo "AUGMENTATION_COPIES must be an integer in [1,8]."
    exit 1
fi

# ── Checkpoint and optimization ────────────────────────────────────────────────
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing conditioned pretraining checkpoint: $PRETRAIN_CKPT"
    exit 1
fi

NUM_CLASSES=5
EPOCHS="${EPOCHS:-100}"
LR="${LR:-$DEFAULT_LR}"
BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.1}"
MIN_LR="${MIN_LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
HEAD_PROJECTION_DIM="${HEAD_PROJECTION_DIM:-64}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-128}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
AMP="${AMP:-true}"
EARLY_STOP="${EARLY_STOP:-false}"
PATIENCE="${PATIENCE:-20}"

# Fixed for every WBC-0903 union experiment.
PATCH_SIZE=16
SPECTRAL_PATCH_SIZE=5
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

# Must match the completed offline NMF run.
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0

WORKERS="${WORKERS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
# Save one validation-best checkpoint per 10-epoch window and test all windows.
BEST_VAL_INTERVAL="${BEST_VAL_INTERVAL:-10}"
PROGRESS="${PROGRESS:-log}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# ── Output naming ──────────────────────────────────────────────────────────────
DATASET_INFO="$(dataset_info_from_roots "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT")"
RUN_TAG="${EXPERIMENT_ID}_${DATASET_INFO}-tok${PATCH_SIZE}-sp${SPECTRAL_PATCH_SIZE}-${CLASSIFICATION_HEAD}-${AUGMENTATION_POLICY}-c${AUGMENTATION_COPIES}-pp${PERSPECTIVE_PROBABILITY}-ps${PERSPECTIVE_SCALE}-${BALANCE_MODE}-lr${LR}-blrm${BACKBONE_LR_MULT}-seed${SEED}"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/test_cls_wbc_0903/${RUN_TAG}_${EXP_TIME}"

echo "============================================================================"
echo "WBC experiment:          $EXPERIMENT_ID"
echo "Train root:              $TRAIN_ROOT"
echo "Val root:                $VAL_ROOT"
echo "Test root:               $TEST_ROOT"
echo "Input/token:             256x256x50 / P${PATCH_SIZE} SP${SPECTRAL_PATCH_SIZE}"
echo "Augment/copies:          $AUGMENT / $AUGMENTATION_COPIES"
echo "Augmentation policy:     $AUGMENTATION_POLICY"
echo "Perspective p/scale/pad: $PERSPECTIVE_PROBABILITY / $PERSPECTIVE_SCALE / $PERSPECTIVE_PADDING_MODE"
echo "Balance mode:            $BALANCE_MODE"
echo "Classification head:     $CLASSIFICATION_HEAD"
echo "LR/backbone multiplier:  $LR / $BACKBONE_LR_MULT"
echo "Seed:                    $SEED"
echo "Save dir:                $SAVE_DIR"
echo "============================================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: configuration and input checks passed; training was not started."
    exit 0
fi
mkdir -p "$SAVE_DIR"

OPTIONAL_ARGS=(--pretrain-ckpt "$PRETRAIN_CKPT" --nmf-simplex)
if [ "$AMP" = "true" ]; then OPTIONAL_ARGS+=(--amp); fi
if [ "$AUGMENT" = "true" ]; then
    OPTIONAL_ARGS+=(--augment)
else
    OPTIONAL_ARGS+=(--no-augment)
fi
if [ "$EARLY_STOP" = "true" ]; then
    OPTIONAL_ARGS+=(--early-stop --patience "$PATIENCE")
fi

MASTER_PORT=$(python3 -c '
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(sock.getsockname()[1])
')

OMP_NUM_THREADS=2 torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    train_finetune_conditioned_cls.py \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT" \
    --test-root "$TEST_ROOT" \
    --num-classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --lr "$LR" \
    --backbone-lr-mult "$BACKBONE_LR_MULT" \
    --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --clip-grad "$CLIP_GRAD" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --class-weighting "$CLASS_WEIGHTING" \
    --sampling-strategy "$SAMPLING_STRATEGY" \
    --augmentation-copies "$AUGMENTATION_COPIES" \
    --augmentation-policy "$AUGMENTATION_POLICY" \
    --perspective-probability "$PERSPECTIVE_PROBABILITY" \
    --perspective-scale "$PERSPECTIVE_SCALE" \
    --perspective-padding-mode "$PERSPECTIVE_PADDING_MODE" \
    --classification-head "$CLASSIFICATION_HEAD" \
    --head-projection-dim "$HEAD_PROJECTION_DIM" \
    --head-hidden-dim "$HEAD_HIDDEN_DIM" \
    --head-dropout "$HEAD_DROPOUT" \
    --seed "$SEED" \
    --patch-size "$PATCH_SIZE" \
    --spectral-patch-size "$SPECTRAL_PATCH_SIZE" \
    --nmf-k "$NMF_K" \
    --nmf-l1 "$NMF_L1" \
    --nmf-l2 "$NMF_L2" \
    --nmf-l3 "$NMF_L3" \
    --nmf-lam-e "$NMF_LAM_E" \
    --nmf-e-clamp-max "$NMF_E_CLAMP_MAX" \
    --embed-dim "$EMBED_DIM" \
    --vit-depth "$VIT_DEPTH" \
    --vit-heads "$VIT_HEADS" \
    --mlp-ratio "$MLP_RATIO" \
    --dropout "$DROPOUT" \
    --cnn-stem-ch "$CNN_STEM_CH" \
    --cnn-spectral-agg "$CNN_SPECTRAL_AGG" \
    --fusion-heads "$FUSION_HEADS" \
    --feature-dim "$FEATURE_DIM" \
    --decoder-mid-ch "$DECODER_MID_CH" \
    --residual-hidden-dim "$RESIDUAL_HIDDEN_DIM" \
    --ridge-lambda "$RIDGE_LAMBDA" \
    --confidence-temperature "$CONFIDENCE_TEMPERATURE" \
    --alpha-min "$ALPHA_MIN" \
    --alpha-extra "$ALPHA_EXTRA" \
    --od-max "$OD_MAX" \
    --workers "$WORKERS" \
    --save-interval "$SAVE_INTERVAL" \
    --best-val-interval "$BEST_VAL_INTERVAL" \
    --progress "$PROGRESS" \
    --log-interval "$LOG_INTERVAL" \
    --save-dir "$SAVE_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$SAVE_DIR/records.txt"
