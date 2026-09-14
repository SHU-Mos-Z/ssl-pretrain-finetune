#!/bin/bash
# CINET backbone 分割微调（须先完成 run_pretrain_cinet.sh）
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# TRAIN_ROOT="data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_train"
# VAL_ROOT="data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_val"
# TEST_ROOT="data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_test"

TRAIN_ROOT="${TRAIN_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_train}"
VAL_ROOT="${VAL_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_val}"
TEST_ROOT="${TEST_ROOT:-data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_test}"


# CINET 预训练 checkpoint（留空则随机初始化）
PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_cinet/20260702_123746/ckpt_epoch0200.pth}"
# PRETRAIN_CKPT="records/pretrain_cinet/20260702_120833/ckpt_epoch0200.pth"

# ── 训练超参数 ────────────────────────────────────────────────────────────────
EPOCHS=100
LR="${LR:-5e-4}"                     # AdamW 峰值学习率
MIN_LR=1e-6                 # 余弦退火最小学习率
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=5             # 线性 warmup epoch 数
CLIP_GRAD=1.0               # 梯度裁剪范数上限
SEED=42
EARLY_STOP=false            # 是否启用 EarlyStopping（基于验证 Dice）
PATIENCE=20                 # EarlyStopping 容忍 epoch 数

# ── 模型架构（须与预训练保持一致） ───────────────────────────────────────────
NUM_CLASSES=2               # 分割类别数
EMBED_DIM=256               # ViT 隐藏维度 D
VIT_DEPTH=6                 # ViT Transformer 层数
VIT_HEADS=8                 # ViT 多头注意力头数
MLP_RATIO=4.0               # ViT FFN 中间维度倍率
DROPOUT=0.1                 # ViT Attention / FFN dropout
PATCH_SIZE=16               # 空间 Patch 边长 P（须与预训练一致）
SPECTRAL_PATCH_SIZE=5      # 谱段组大小 s_p（须与预训练一致）
NUM_ENDMEMBERS=16            # NMF 端元数 K（须与预训练一致）
AGGREGATE_MODE="mean"       # 跨谱段聚合方式：mean | attention

# ── CNN 路径超参数（须与预训练保持一致） ─────────────────────────────────────
CNN_STEM_CH=64              # Stem 输出通道数
CNN_SPECTRAL_AGG="attention"  # 波段聚合方式：mean | max | attention

# ── CIAM 超参数（须与预训练保持一致） ────────────────────────────────────────
CIAM_HEADS=8                # CIAM 多头注意力头数
CIAM_DROPOUT=0.1            # CIAM attention / FFN dropout
CIAM_FFN_RATIO=2.0          # CIAM FFN 中间维度倍率

# ── PixelDecoder 超参数 ───────────────────────────────────────────────────────
DECODER_MID_CH=64           # PixelDecoder final_conv 中间通道数
                            # 注：微调时 out_ch=NUM_CLASSES，不从 checkpoint 加载此层

# ── 迁移学习 ──────────────────────────────────────────────────────────────────
FREEZE_BACKBONE=false       # 是否冻结 backbone（Linear Probe 模式设为 true）
# 留空时不传参，由 Python 使用历史 all_class 损失；设为 foreground
# 时使用全类别加权 CE + 前景 Dice + 前景 boundary。
SEGMENTATION_LOSS_MODE="${SEGMENTATION_LOSS_MODE:-}"

# ── 日志 / 存储 ───────────────────────────────────────────────────────────────
WORKERS=4
SAVE_INTERVAL=10
N_VIS=8
PROGRESS="log"
EXP_TIME=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./records/finetune_cinet/${EXP_TIME}"
mkdir -p "$SAVE_DIR"

# 可选参数组合
PRETRAIN_ARG=""
if [ -n "$PRETRAIN_CKPT" ]; then
    PRETRAIN_ARG="--pretrain-ckpt $PRETRAIN_CKPT"
fi
EARLY_STOP_ARG=""
if [ "$EARLY_STOP" = "true" ]; then
    EARLY_STOP_ARG="--early-stop --patience $PATIENCE"
fi
FREEZE_ARG=""
if [ "$FREEZE_BACKBONE" = "true" ]; then
    FREEZE_ARG="--freeze-backbone"
fi
SEGMENTATION_LOSS_MODE_ARGS=()
if [ -n "$SEGMENTATION_LOSS_MODE" ]; then
    SEGMENTATION_LOSS_MODE_ARGS+=(--segmentation-loss-mode "$SEGMENTATION_LOSS_MODE")
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
    train_finetune_cinet.py \
    --train-root          "$TRAIN_ROOT" \
    --val-root            "$VAL_ROOT" \
    --test-root           "$TEST_ROOT" \
    --epochs              $EPOCHS \
    --batch-size          $BATCH_SIZE_PER_GPU \
    --lr                  $LR \
    --min-lr              $MIN_LR \
    --weight-decay        $WEIGHT_DECAY \
    --warmup-epochs       $WARMUP_EPOCHS \
    --clip-grad           $CLIP_GRAD \
    --seed                $SEED \
    --num-classes         $NUM_CLASSES \
    --embed-dim           $EMBED_DIM \
    --vit-depth           $VIT_DEPTH \
    --vit-heads           $VIT_HEADS \
    --mlp-ratio           $MLP_RATIO \
    --dropout             $DROPOUT \
    --patch-size          $PATCH_SIZE \
    --spectral-patch-size $SPECTRAL_PATCH_SIZE \
    --num-endmembers      $NUM_ENDMEMBERS \
    --aggregate-mode      $AGGREGATE_MODE \
    --cnn-stem-ch         $CNN_STEM_CH \
    --cnn-spectral-agg    $CNN_SPECTRAL_AGG \
    --ciam-heads          $CIAM_HEADS \
    --ciam-dropout        $CIAM_DROPOUT \
    --ciam-ffn-ratio      $CIAM_FFN_RATIO \
    --decoder-mid-ch      $DECODER_MID_CH \
    "${SEGMENTATION_LOSS_MODE_ARGS[@]}" \
    --workers             $WORKERS \
    --save-interval       $SAVE_INTERVAL \
    --n-vis               $N_VIS \
    --progress            $PROGRESS \
    --save-dir            "$SAVE_DIR" \
    --amp \
    $PRETRAIN_ARG \
    $EARLY_STOP_ARG \
    $FREEZE_ARG \
    2>&1 | tee "$SAVE_DIR/records.txt"
