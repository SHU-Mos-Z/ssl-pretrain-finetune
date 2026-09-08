#!/bin/bash
# 逐图端元条件化 + 残差丰度约束预训练
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4

DATA_ROOTS=(
    "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_pretrain_p030_20260728"
    "data/2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50_pretrain_p030_20260728"
    "data/LUAD_patch_400x400_overlap_0x0_to_256x256_minmax/Training_pretrain_p030_20260728"
    "data/PDAC_pretrain_preprocessed_pretrain_p100_20260728"
)

# ── Token 化 / 混合遮蔽 ─────────────────────────────────────────────────────
PATCH_SIZE=16
SPECTRAL_PATCH_SIZE=5
SPECTRAL_MASK_RATIO=0.30
SPATIAL_MASK_RATIO=0.20
ALLOW_INDEX_WAVELENGTHS=true  # 有 wavelengths.npy 后设为 false
PAD_TO_PATCH=true
PERMUTE_ENDMEMBERS=true

# ── 离线 NMF 缓存键（须与 run_offline_nmf.sh 全部参数保持一致） ──────────────
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true
NMF_LAM_E=0.05
NMF_E_CLAMP_MAX=3.0
NMF_WEIGHT_TEMPERATURE=0.05

# ── 训练超参数 ────────────────────────────────────────────────────────────────
EPOCHS=200
LR=1e-4
MIN_LR=2e-6
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=10
CLIP_GRAD=1.0
SEED=42
AMP=true

# ── ViT / Token 路径超参数 ────────────────────────────────────────────────────
EMBED_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
MLP_RATIO=4.0
DROPOUT=0.1

# ── CNN 路径与置信融合超参数 ──────────────────────────────────────────────────
CNN_STEM_CH=64
CNN_SPECTRAL_AGG="attention"
FUSION_HEADS=8

# ── FeatureDecoder / 残差丰度头超参数 ────────────────────────────────────────
FEATURE_DIM=128
DECODER_MID_CH=64
RESIDUAL_HIDDEN_DIM=128
RIDGE_LAMBDA=1e-3
CONFIDENCE_TEMPERATURE=0.05
ALPHA_MIN=0.1
ALPHA_EXTRA=1.0
OD_MAX=3.0

# ── Pretext 损失权重 ──────────────────────────────────────────────────────────
LAMBDA_OD=1.0
LAMBDA_I=1.0
LAMBDA_C=0.2
LAMBDA_TOKEN=1.0
LAMBDA_FEATURE=0.0       # >0 时自动执行双 Mask 前向
LAMBDA_DELTA=0.01
LAMBDA_SAM=0.0

# ── 日志 / 存储 ────────────────────────────────────────────────────────────────
WORKERS=4
SAVE_INTERVAL=10
PROGRESS="log"
LOG_INTERVAL=20
RESUME=""
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/pretrain_conditioned/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

# ── 可选参数组合 ──────────────────────────────────────────────────────────────
WAVELENGTH_ARG=""
if [ "$ALLOW_INDEX_WAVELENGTHS" = "true" ]; then
    WAVELENGTH_ARG="--allow-index-wavelengths"
fi
PAD_ARG="--no-pad-to-patch"
if [ "$PAD_TO_PATCH" = "true" ]; then
    PAD_ARG="--pad-to-patch"
fi
PERMUTE_ARG="--no-permute-endmembers"
if [ "$PERMUTE_ENDMEMBERS" = "true" ]; then
    PERMUTE_ARG="--permute-endmembers"
fi
SIMPLEX_ARG="--no-nmf-simplex"
if [ "$NMF_SIMPLEX" = "true" ]; then
    SIMPLEX_ARG="--nmf-simplex"
fi
AMP_ARG=""
if [ "$AMP" = "true" ]; then
    AMP_ARG="--amp"
fi
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

MASTER_PORT=$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
")

OMP_NUM_THREADS=2 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_pretrain_conditioned.py \
    --root                   "${DATA_ROOTS[@]}" \
    --patch-size             $PATCH_SIZE \
    --spectral-patch-size    $SPECTRAL_PATCH_SIZE \
    --spectral-mask-ratio    $SPECTRAL_MASK_RATIO \
    --spatial-mask-ratio     $SPATIAL_MASK_RATIO \
    --nmf-k                  $NMF_K \
    --nmf-l1                 $NMF_L1 \
    --nmf-l2                 $NMF_L2 \
    --nmf-l3                 $NMF_L3 \
    --nmf-lam-e              $NMF_LAM_E \
    --nmf-e-clamp-max        $NMF_E_CLAMP_MAX \
    --nmf-weight-temperature $NMF_WEIGHT_TEMPERATURE \
    --epochs                 $EPOCHS \
    --batch-size             $BATCH_SIZE_PER_GPU \
    --lr                     $LR \
    --min-lr                 $MIN_LR \
    --weight-decay           $WEIGHT_DECAY \
    --warmup-epochs          $WARMUP_EPOCHS \
    --clip-grad              $CLIP_GRAD \
    --seed                   $SEED \
    --embed-dim              $EMBED_DIM \
    --vit-depth              $VIT_DEPTH \
    --vit-heads              $VIT_HEADS \
    --mlp-ratio              $MLP_RATIO \
    --dropout                $DROPOUT \
    --cnn-stem-ch            $CNN_STEM_CH \
    --cnn-spectral-agg       $CNN_SPECTRAL_AGG \
    --fusion-heads           $FUSION_HEADS \
    --feature-dim            $FEATURE_DIM \
    --decoder-mid-ch         $DECODER_MID_CH \
    --residual-hidden-dim    $RESIDUAL_HIDDEN_DIM \
    --ridge-lambda           $RIDGE_LAMBDA \
    --confidence-temperature $CONFIDENCE_TEMPERATURE \
    --alpha-min              $ALPHA_MIN \
    --alpha-extra            $ALPHA_EXTRA \
    --od-max                 $OD_MAX \
    --lambda-od              $LAMBDA_OD \
    --lambda-i               $LAMBDA_I \
    --lambda-c               $LAMBDA_C \
    --lambda-token           $LAMBDA_TOKEN \
    --lambda-feature         $LAMBDA_FEATURE \
    --lambda-delta           $LAMBDA_DELTA \
    --lambda-sam             $LAMBDA_SAM \
    --workers                $WORKERS \
    --save-interval          $SAVE_INTERVAL \
    --progress               $PROGRESS \
    --log-interval           $LOG_INTERVAL \
    --save-dir               "$SAVE_DIR" \
    $WAVELENGTH_ARG \
    $PAD_ARG \
    $PERMUTE_ARG \
    $SIMPLEX_ARG \
    $AMP_ARG \
    $RESUME_ARG \
    2>&1 | tee "$SAVE_DIR/records.txt"
