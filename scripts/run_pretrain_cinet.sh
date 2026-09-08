#!/bin/bash
# NMF 前置 + CINET（CNN × ViT × CIAM）Pretext 预训练
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NUM_GPUS=2
BATCH_SIZE_PER_GPU=4

DATA_ROOTS=(
    "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed"
    # data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed
)

# ── Token 化 / 掩膜 ─────────────────────────────────────────────────────────
PATCH_SIZE=16               # 空间 Patch 边长 P（须为 2 的整数次幂，对应 ContextualEncoder spatial_depth）
SPECTRAL_PATCH_SIZE=5      # 谱段组大小 s_p
MASK_RATIO=0.4              # Token 遮蔽比例
USE_GRADIENT_MASKING=true   # 是否启用梯度引导 masking
SOBEL_TAU=1.0               # 梯度 masking 空间权重温度系数
SPECTRAL_ALPHA=1.0          # 梯度 masking 谱段权重平衡系数

# ── 离线 NMF 缓存键（须与 run_offline_nmf.sh 全部参数保持一致） ──────────────
NMF_K=16
NMF_L1=5e-4
NMF_L2=2e-4
NMF_L3=1e-2
NMF_SIMPLEX=true            # true=读取 _simplex 后缀缓存
NMF_LAM_E=0.05              # E L2 正则强度（对应 run_offline_nmf.sh LAM_E）
NMF_E_CLAMP_MAX=3.0         # E 逐元素上界（对应 E_CLAMP_MAX；0 表示未启用）

# ── 训练超参数 ────────────────────────────────────────────────────────────────
EPOCHS=200
LR=4e-4                     # AdamW 峰值学习率
MIN_LR=2e-6                 # 余弦退火最小学习率
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=20             # 线性 warmup epoch 数（测试）
CLIP_GRAD=1.0               # 梯度裁剪范数上限
SEED=42

# ── ViT / Token 路径超参数 ────────────────────────────────────────────────────
EMBED_DIM=256               # ViT 隐藏维度 D
VIT_DEPTH=6                 # ViT Transformer 层数
VIT_HEADS=8                 # ViT 多头注意力头数
MLP_RATIO=4.0               # ViT FFN 中间维度倍率
DROPOUT=0.1                 # ViT Attention / FFN dropout
NUM_ENDMEMBERS=16            # NMF 端元数 K（须与 NMF_K 一致）
AGGREGATE_MODE="mean"       # 跨谱段聚合方式：mean | attention
ABUNDANCE_ACT="softmax"     # 丰度激活函数：softmax | softplus
OD_MAX=3.0                  # OD clamp 上限（数值稳定）

# ── CNN 路径超参数（ContextualEncoder + SpectralAggregator） ─────────────────
CNN_STEM_CH=64              # Stem 输出通道数（后续每层翻倍，共 spatial_depth 层）
CNN_SPECTRAL_AGG="attention"  # 波段聚合方式：mean | max | attention

# ── CIAM 交叉注意力超参数 ─────────────────────────────────────────────────────
CIAM_HEADS=8                # CIAM 多头注意力头数（须能整除 EMBED_DIM）
CIAM_DROPOUT=0.1            # CIAM attention / FFN dropout
CIAM_FFN_RATIO=2.0          # CIAM FFN 中间维度倍率

# ── PixelDecoder 超参数 ───────────────────────────────────────────────────────
DECODER_MID_CH=64           # PixelDecoder final_conv 中间通道数

# ── 损失权重 ──────────────────────────────────────────────────────────────────
LAMBDA_OD=1.0               # OD 重建损失权重 λ_OD
LAMBDA_I=1.0                # 强度重建损失权重 λ_I
LAMBDA_CONS_PIX=0.2         # 像素级一致性损失权重 λ_pix

# ── Token 级一致性损失开关（DINO-style） ──────────────────────────────────────
# USE_CONS_TOKEN=true  → 启用 l_cons_token + l_anchor
# USE_CONS_TOKEN=false → 完全关闭，仅用 od/i/cons_pix 训练
USE_CONS_TOKEN=false
LAMBDA_CONS_TOKEN=0.2       # Token 级一致性损失权重 λ_tok（USE_CONS_TOKEN=true 时生效）
LAMBDA_ANCHOR=0.1           # teacher_proj NMF 锚定损失权重（0=不启用锚定）
PROJ_DIM=128                # TokenConsistencyHead 投影维度 D_L

# ── 日志 / 存储 ───────────────────────────────────────────────────────────────
WORKERS=4
SAVE_INTERVAL=10
PROGRESS="log"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/pretrain_cinet/${EXP_TIME}"
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
    train_pretrain_cinet.py \
    --root                "${DATA_ROOTS[@]}" \
    --patch-size          $PATCH_SIZE \
    --spectral-patch-size $SPECTRAL_PATCH_SIZE \
    --mask-ratio          $MASK_RATIO \
    $([ "$USE_GRADIENT_MASKING" = "true" ] && echo "--use-gradient-masking") \
    --sobel-tau           $SOBEL_TAU \
    --spectral-alpha      $SPECTRAL_ALPHA \
    --nmf-k               $NMF_K \
    --nmf-l1              $NMF_L1 \
    --nmf-l2              $NMF_L2 \
    --nmf-l3              $NMF_L3 \
    $([ "$NMF_SIMPLEX" = "true" ] && echo "--nmf-simplex") \
    --nmf-lam-e           $NMF_LAM_E \
    --nmf-e-clamp-max     $NMF_E_CLAMP_MAX \
    --epochs              $EPOCHS \
    --batch-size          $BATCH_SIZE_PER_GPU \
    --lr                  $LR \
    --min-lr              $MIN_LR \
    --weight-decay        $WEIGHT_DECAY \
    --warmup-epochs       $WARMUP_EPOCHS \
    --clip-grad           $CLIP_GRAD \
    --seed                $SEED \
    --embed-dim           $EMBED_DIM \
    --vit-depth           $VIT_DEPTH \
    --vit-heads           $VIT_HEADS \
    --mlp-ratio           $MLP_RATIO \
    --dropout             $DROPOUT \
    --num-endmembers      $NUM_ENDMEMBERS \
    --aggregate-mode      $AGGREGATE_MODE \
    --abundance-act       $ABUNDANCE_ACT \
    --od-max              $OD_MAX \
    --cnn-stem-ch         $CNN_STEM_CH \
    --cnn-spectral-agg    $CNN_SPECTRAL_AGG \
    --ciam-heads          $CIAM_HEADS \
    --ciam-dropout        $CIAM_DROPOUT \
    --ciam-ffn-ratio      $CIAM_FFN_RATIO \
    --decoder-mid-ch      $DECODER_MID_CH \
    --lambda-od           $LAMBDA_OD \
    --lambda-i            $LAMBDA_I \
    --lambda-cons-pix     $LAMBDA_CONS_PIX \
    --lambda-cons-token   $LAMBDA_CONS_TOKEN \
    --lambda-anchor       $LAMBDA_ANCHOR \
    $([ "$USE_CONS_TOKEN" = "true" ] && echo "--use-cons-token" || echo "--no-use-cons-token") \
    --proj-dim            $PROJ_DIM \
    --workers             $WORKERS \
    --save-interval       $SAVE_INTERVAL \
    --progress            $PROGRESS \
    --save-dir            "$SAVE_DIR" \
    --amp \
    2>&1 | tee "$SAVE_DIR/records.txt"
