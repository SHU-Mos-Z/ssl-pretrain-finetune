"""
CINET 预训练权重 → 分割微调入口。

启动：
    OMP_NUM_THREADS=2 torchrun --nproc_per_node=<N> train_finetune_cinet.py [args...]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.finetune_model_cinet import FinetuneModelCINET
from utils.datasets import build_finetune_loaders
from utils.losses import SegLoss
from utils.metrics import dice_score, hd95_score, iou_score
from utils.scheduler import build_cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="CINET backbone 分割微调")

    # ── 数据 ──
    p.add_argument("--train-root", required=True,
                   help="训练集根目录（含 images/ 与 masks/）")
    p.add_argument("--val-root", required=True,
                   help="验证集根目录")
    p.add_argument("--test-root", default=None,
                   help="测试集根目录（可选，用于最终评估）")

    # ── 训练 ──
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="峰值学习率（AdamW）")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="余弦退火最小学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="线性 warmup 的 epoch 数")
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="梯度裁剪范数上限（0 = 不裁剪）")
    p.add_argument("--amp", action="store_true",
                   help="启用混合精度训练（AMP）")
    p.add_argument("--early-stop", action="store_true",
                   help="启用 EarlyStopping（基于验证 Dice）")
    p.add_argument("--patience", type=int, default=20,
                   help="EarlyStopping 容忍 epoch 数")

    # ── 模型 ──
    p.add_argument("--num-classes", type=int, default=2,
                   help="分割类别数")
    p.add_argument("--embed-dim", type=int, default=256,
                   help="ViT 隐藏维度 D")
    p.add_argument("--vit-depth", type=int, default=6,
                   help="ViT Transformer 层数")
    p.add_argument("--vit-heads", type=int, default=8,
                   help="ViT 多头注意力头数")
    p.add_argument("--mlp-ratio", type=float, default=4.0,
                   help="ViT FFN 中间维度倍率")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="ViT Attention / FFN dropout")
    p.add_argument("--patch-size", type=int, default=16,
                   help="空间 Patch 边长 P（须与预训练一致）")
    p.add_argument("--spectral-patch-size", type=int, default=10,
                   help="谱段组大小 s_p（须与预训练一致）")
    p.add_argument("--num-endmembers", type=int, default=2,
                   help="NMF 端元数 K（须与预训练一致）")
    p.add_argument("--aggregate-mode", default="mean",
                   choices=["mean", "attention"],
                   help="跨谱段聚合方式")

    # ── CNN 路径超参数（须与预训练一致） ──
    p.add_argument("--cnn-stem-ch", type=int, default=64,
                   help="ContextualEncoder Stem 输出通道数")
    p.add_argument("--cnn-spectral-agg", default="attention",
                   choices=["mean", "max", "attention"],
                   help="SpectralAggregator 波段聚合方式")

    # ── CIAM 超参数（须与预训练一致） ──
    p.add_argument("--ciam-heads", type=int, default=8,
                   help="CIAM 交叉注意力多头数")
    p.add_argument("--ciam-dropout", type=float, default=0.1,
                   help="CIAM attention / FFN dropout")
    p.add_argument("--ciam-ffn-ratio", type=float, default=2.0,
                   help="CIAM FFN 中间维度倍率")

    # ── PixelDecoder 超参数 ──
    p.add_argument("--decoder-mid-ch", type=int, default=64,
                   help="PixelDecoder final_conv 中间通道数")

    # ── 迁移学习 ──
    p.add_argument("--pretrain-ckpt", default=None,
                   help="NMFPretrainModelCINET checkpoint 路径（可选）")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="冻结 backbone，仅训练 PixelDecoder（Linear Probe）")

    # ── 日志 / 存储 ──
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="records/finetune_cinet/run")
    p.add_argument("--save-interval", type=int, default=10,
                   help="每隔多少 epoch 保存一次 checkpoint")
    p.add_argument("--progress", default="tqdm",
                   choices=["tqdm", "log", "none"])
    p.add_argument("--n-vis", type=int, default=8,
                   help="测试集可视化保存数量")
    return p.parse_args()


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_checkpoint(model: nn.Module, path: str) -> None:
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(state, path)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    scaler, epoch, args, sampler=None,
):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total, n = 0.0, 0
    iterable = enumerate(loader)
    if args.progress == "tqdm" and is_main():
        iterable = tqdm(iterable, total=len(loader),
                        desc=f"Train {epoch:04d}", leave=False)
    for _, batch in iterable:
        batch = _move_batch(batch, torch.cuda.current_device())
        seg = batch.pop("seg")
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            logits = model(batch, w_abund=0.0)
            loss, _ = criterion(logits, seg)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        scheduler.step()
        total += loss.item() * seg.shape[0]
        n += seg.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, num_classes, args):
    model.eval()
    dice_vals, iou_vals = [], []
    all_preds, all_targets = [], []
    for batch in loader:
        batch = _move_batch(batch, torch.cuda.current_device())
        seg = batch.pop("seg")
        logits = model(batch, w_abund=0.0)
        dice_vals.append(dice_score(logits, seg, num_classes).item())
        iou_vals.append(iou_score(logits, seg, num_classes).item())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(seg.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return (
        float(np.mean(dice_vals)),
        float(np.mean(iou_vals)),
        hd95_score(all_preds, all_targets, num_classes),
    )


@torch.no_grad()
def save_predictions(model, loader, save_dir, n_vis, num_classes):
    model.eval()
    preds_dir = os.path.join(save_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    count = 0
    for batch in loader:
        if count >= n_vis:
            break
        batch_cpu = dict(batch)
        seg = batch_cpu.pop("seg")
        dev_batch = _move_batch(batch_cpu, torch.cuda.current_device())
        logits = model(dev_batch, w_abund=0.0)
        pred = logits.argmax(dim=1).cpu().numpy()
        seg_np = seg.numpy()
        for b in range(min(pred.shape[0], n_vis - count)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(seg_np[b], cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[0].set_title("GT")
            axes[1].imshow(pred[b], cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[1].set_title("Pred")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(preds_dir, f"sample_{count:04d}.png"), dpi=100)
            plt.close(fig)
            count += 1
    log(f"预测可视化已保存至 {preds_dir}（{count} 张）")


def main():
    args = get_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    fix_seed(args.seed + dist.get_rank())
    os.makedirs(args.save_dir, exist_ok=True)

    if is_main():
        print(
            "=" * 60,
            "\n".join(f"  {k}: {v}" for k, v in vars(args).items()),
            "=" * 60, sep="\n", flush=True,
        )

    # CINET 微调需要在 batch 中包含完整 OD 立方
    train_loader, val_loader, test_loader, train_sampler = build_finetune_loaders(
        train_root=args.train_root,
        val_root=args.val_root,
        test_root=args.test_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        distributed=True,
        include_od=True,
    )
    log(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    # 推断 n_sp
    sample = train_loader.dataset[0]
    h_p = sample["H"] // args.patch_size
    w_p = sample["W"] // args.patch_size
    n_sp = sample["token_raw"].shape[0] // (h_p * w_p)

    model = FinetuneModelCINET(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        spectral_patch_size=args.spectral_patch_size,
        num_endmembers=args.num_endmembers,
        num_spectral_groups=n_sp,
        patch_size=args.patch_size,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_abund_pe=False,
        aggregate_mode=args.aggregate_mode,
        cnn_stem_ch=args.cnn_stem_ch,
        cnn_spectral_agg=args.cnn_spectral_agg,
        ciam_heads=args.ciam_heads,
        ciam_dropout=args.ciam_dropout,
        ciam_ffn_ratio=args.ciam_ffn_ratio,
        decoder_mid_ch=args.decoder_mid_ch,
        pretrain_ckpt=None,
        freeze_backbone=args.freeze_backbone,
    ).cuda()

    if args.pretrain_ckpt:
        model.load_pretrain(args.pretrain_ckpt)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {total_params / 1e6:.2f}M  n_sp={n_sp}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = build_cosine_scheduler(
        optimizer, args.epochs, args.warmup_epochs,
        len(train_loader), args.lr, args.min_lr,
    )
    criterion = SegLoss(num_classes=args.num_classes)
    scaler = GradScaler(enabled=args.amp)

    best_dice, no_improve = 0.0, 0
    best_ckpt = os.path.join(args.save_dir, "ckpt_best.pth")
    stop_signal = torch.zeros(1, dtype=torch.int32, device=f"cuda:{local_rank}")

    log("开始 CINET 微调")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, epoch, args, sampler=train_sampler,
        )
        val_dice, val_iou, val_hd95 = evaluate(
            model, val_loader, args.num_classes, args,
        )
        stop_signal.fill_(0)
        if is_main():
            log(
                f"Epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  "
                f"val_hd95={val_hd95:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"time={time.time()-t0:.1f}s"
            )
            if val_dice > best_dice:
                best_dice = val_dice
                no_improve = 0
                save_checkpoint(model, best_ckpt)
                log(f"  ★ 最优 Dice={best_dice:.4f} → {best_ckpt}")
            else:
                no_improve += 1
                if args.early_stop and no_improve >= args.patience:
                    log(f"Early Stop @ epoch {epoch}")
                    stop_signal.fill_(1)
            if epoch % args.save_interval == 0:
                save_checkpoint(
                    model,
                    os.path.join(args.save_dir, f"ckpt_epoch{epoch:04d}.pth"),
                )
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    if is_main() and test_loader is not None and os.path.isfile(best_ckpt):
        inner = model.module if isinstance(model, DDP) else model
        inner.load_state_dict(
            torch.load(best_ckpt, map_location="cpu", weights_only=False),
        )
        test_dice, test_iou, test_hd95 = evaluate(
            model, test_loader, args.num_classes, args,
        )
        log(f"测试集结果  Dice={test_dice:.4f}  IoU={test_iou:.4f}  HD95={test_hd95:.4f}")
        save_predictions(model, test_loader, args.save_dir, args.n_vis, args.num_classes)

    dist.destroy_process_group()
    log("CINET 微调完成")


if __name__ == "__main__":
    main()
