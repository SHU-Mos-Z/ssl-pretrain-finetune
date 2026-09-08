#!/bin/bash
# NMF 前置 + ViT Pretext 预训练（无 SLIC/MUR）
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
BATCH_SIZE_PER_GPU=4

DATA_ROOTS=(
    "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed"
)

# Token 化 / 掩膜
PATCH_SIZE=16
SPECTRAL_PATCH_SIZE=10
MASK_RATIO=0.4
USE_GRADIENT_MASKING=true
SOBEL_TAU=1.0
SPECTRAL_ALPHA=1.0

# 离线 NMF 缓存键（须与 run_offline_nmf.sh 全部参数保持一致）
NMF_K=8
NMF_L1=1e-3
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true            # true=读取 _simplex 后缀缓存
NMF_LAM_E=0.05              # E L2 正则强度（对应 run_offline_nmf.sh LAM_E）
NMF_E_CLAMP_MAX=3.0         # E 逐元素上界（对应 E_CLAMP_MAX；0 表示未启用）

# 训练
EPOCHS=200
LR=4e-4
MIN_LR=2e-6
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=10
CLIP_GRAD=1.0
SEED=42

# 模型
EMBED_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
NUM_ENDMEMBERS=8
AGGREGATE_MODE="mean"
ABUNDANCE_ACT="softmax"
USE_REFINE=true
OD_MAX=3.0

# 损失
LAMBDA_OD=1.0
LAMBDA_I=1.0
LAMBDA_CONS_PIX=0.2

# ── Token 级一致性损失开关（DINO-style） ──────────────────────────────────────
# USE_CONS_TOKEN=true  → 启用 l_cons_token + l_anchor
# USE_CONS_TOKEN=false → 完全关闭，仅用 od/i/cons_pix 训练
USE_CONS_TOKEN=true
LAMBDA_CONS_TOKEN=0.2       # Token 级一致性损失权重（USE_CONS_TOKEN=true 时生效）
LAMBDA_ANCHOR=0.1           # teacher_proj NMF 锚定损失权重（0=不启用锚定）
PROJ_DIM=128                # TokenConsistencyHead 投影维度 D_L

WORKERS=4
SAVE_INTERVAL=20
PROGRESS="log"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/pretrain/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

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
    train_pretrain_vit.py \
    --root               "${DATA_ROOTS[@]}" \
    --patch-size         $PATCH_SIZE \
    --spectral-patch-size $SPECTRAL_PATCH_SIZE \
    --mask-ratio         $MASK_RATIO \
    $([ "$USE_GRADIENT_MASKING" = "true" ] && echo "--use-gradient-masking") \
    --sobel-tau          $SOBEL_TAU \
    --spectral-alpha     $SPECTRAL_ALPHA \
    --nmf-k              $NMF_K \
    --nmf-l1             $NMF_L1 \
    --nmf-l2             $NMF_L2 \
    --nmf-l3             $NMF_L3 \
    $([ "$NMF_SIMPLEX" = "true" ] && echo "--nmf-simplex") \
    --nmf-lam-e          $NMF_LAM_E \
    --nmf-e-clamp-max    $NMF_E_CLAMP_MAX \
    --epochs             $EPOCHS \
    --batch-size         $BATCH_SIZE_PER_GPU \
    --lr                 $LR \
    --min-lr             $MIN_LR \
    --weight-decay       $WEIGHT_DECAY \
    --warmup-epochs      $WARMUP_EPOCHS \
    --clip-grad          $CLIP_GRAD \
    --seed               $SEED \
    --embed-dim          $EMBED_DIM \
    --vit-depth          $VIT_DEPTH \
    --vit-heads          $VIT_HEADS \
    --num-endmembers     $NUM_ENDMEMBERS \
    --aggregate-mode     $AGGREGATE_MODE \
    --abundance-act      $ABUNDANCE_ACT \
    $([ "$USE_REFINE" = "true" ] && echo "--use-refine" || echo "--no-use-refine") \
    --od-max             $OD_MAX \
    --lambda-od          $LAMBDA_OD \
    --lambda-i           $LAMBDA_I \
    --lambda-cons-pix    $LAMBDA_CONS_PIX \
    --lambda-cons-token  $LAMBDA_CONS_TOKEN \
    --lambda-anchor      $LAMBDA_ANCHOR \
    $([ "$USE_CONS_TOKEN" = "true" ] && echo "--use-cons-token" || echo "--no-use-cons-token") \
    --proj-dim           $PROJ_DIM \
    --workers            $WORKERS \
    --save-interval      $SAVE_INTERVAL \
    --progress           $PROGRESS \
    --save-dir           "$SAVE_DIR" \
    --amp \
    2>&1 | tee "$SAVE_DIR/records.txt"
