"""
NMF 前置 + ViT 四重 Pretext 预训练入口。

启动：
    OMP_NUM_THREADS=2 torchrun --nproc_per_node=<N> train_pretrain_vit.py [args...]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.nmf_pretrain_model_vit import NMFPretrainModel
from utils.datasets import build_pretrain_loader, set_dataset_epoch
from utils.losses import NMFPretextLoss
from utils.pretrain_loss_plotter import PretrainLossPlotter
from utils.scheduler import build_cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="NMF 前置 ViT Pretext 预训练")

    # 数据
    p.add_argument(
        "--root",
        nargs="+",
        required=True,
        help="数据根目录（含 images/ 与 nmf_cache_*）",
    )
    p.add_argument("--patch-size", type=int, default=16, help="空间 Patch 边长 P")
    p.add_argument("--spectral-patch-size", type=int, default=10, help="谱段组大小 s_p")
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--use-gradient-masking", action="store_true")
    p.add_argument("--sobel-tau", type=float, default=1.0)
    p.add_argument("--spectral-alpha", type=float, default=1.0)

    # NMF 缓存
    p.add_argument("--nmf-k", type=int, default=2)
    p.add_argument("--nmf-l1", type=float, default=1e-3)
    p.add_argument("--nmf-l2", type=float, default=1e-4)
    p.add_argument("--nmf-l3", type=float, default=1e-2)
    p.add_argument(
        "--nmf-simplex",
        action="store_true",
        help="读取带 simplex 约束的 NMF 缓存（目录名含 _simplex 后缀）",
    )
    p.add_argument(
        "--nmf-lam-e",
        type=float,
        default=0.0,
        help="NMF 时 E 的 L2 正则强度（须与 run_offline_nmf.sh 的 LAM_E 一致）",
    )
    p.add_argument(
        "--nmf-e-clamp-max",
        type=float,
        default=0.0,
        help="NMF 时 E 的逐元素上界（须与 E_CLAMP_MAX 一致；0 表示未启用）",
    )

    # 训练
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")

    # 模型
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--vit-depth", type=int, default=6)
    p.add_argument("--vit-heads", type=int, default=8)
    p.add_argument("--num-endmembers", type=int, default=2)
    p.add_argument("--aggregate-mode", default="mean", choices=["mean", "attention"])
    p.add_argument(
        "--abundance-act", default="softmax", choices=["softmax", "softplus"]
    )
    p.add_argument("--use-refine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--od-max", type=float, default=3.0)

    # 损失
    p.add_argument("--lambda-od", type=float, default=1.0)
    p.add_argument("--lambda-i", type=float, default=1.0)
    p.add_argument("--lambda-cons-pix", type=float, default=0.5)
    p.add_argument("--lambda-cons-token", type=float, default=0.5)
    p.add_argument(
        "--use-cons-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 l_cons_token + l_anchor（False 时完全关闭）",
    )
    p.add_argument(
        "--lambda-anchor",
        type=float,
        default=0.1,
        help="teacher_proj 锚定损失权重 λ_anchor（0 表示不启用）",
    )
    p.add_argument(
        "--proj-dim",
        type=int,
        default=128,
        help="TokenConsistencyHead 投影维度 D_L（须与模型保持一致）",
    )

    # 其他
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="records/pretrain/run")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--progress", default="tqdm", choices=["tqdm", "log", "none"])
    p.add_argument(
        "--skip-nonfinite-loss", action=argparse.BooleanOptionalAction, default=True
    )
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


def save_checkpoint(model: nn.Module, epoch: int, save_dir: str, tag: str = "") -> None:
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    path = os.path.join(save_dir, f"ckpt_epoch{epoch:04d}{tag}.pth")
    torch.save(state, path)
    log(f"Checkpoint 已保存: {path}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler, epoch, args, sampler=None
):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    set_dataset_epoch(loader, epoch)

    keys = ["loss_od", "loss_i", "loss_cons_pix", "loss_cons_token", "loss_anchor"]
    total_loss = 0.0
    loss_accum = {k: 0.0 for k in keys}
    n_samples = 0
    skipped = 0

    iterable = enumerate(loader)
    if args.progress == "tqdm" and is_main():
        iterable = tqdm(
            iterable, total=len(loader), desc=f"Epoch {epoch:04d}", leave=False
        )

    for step, batch in iterable:
        batch = _move_batch(batch, torch.cuda.current_device())
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            out = model(batch)
            loss, ld = criterion(out, batch)

        if args.skip_nonfinite_loss and not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

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

        bs = batch["token_raw"].shape[0]
        total_loss += ld["loss_total"] * bs
        for k in keys:
            loss_accum[k] += ld[k] * bs
        n_samples += bs

        if args.progress == "log" and is_main() and (step + 1) % 20 == 0:
            log(f"  step {step+1}/{len(loader)}  loss={ld['loss_total']:.4f}")

    n = max(n_samples, 1)
    return {
        "loss": total_loss / n,
        **{k: loss_accum[k] / n for k in keys},
        "skipped_nonfinite": skipped,
    }


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
            "\n  超参数\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items()),
            "=" * 60,
            sep="\n",
            flush=True,
        )

    n_sp = None  # derived from data S / s_p inside dataset
    loader, sampler = build_pretrain_loader(
        data_roots=args.root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        mask_ratio=args.mask_ratio,
        use_gradient_masking=args.use_gradient_masking,
        sobel_tau=args.sobel_tau,
        spectral_alpha=args.spectral_alpha,
        nmf_k=args.nmf_k,
        nmf_l1=args.nmf_l1,
        nmf_l2=args.nmf_l2,
        nmf_l3=args.nmf_l3,
        nmf_simplex=args.nmf_simplex,
        nmf_lam_e=args.nmf_lam_e,
        nmf_e_clamp_max=args.nmf_e_clamp_max,
        total_epochs=args.epochs,
        distributed=True,
        seed=args.seed,
    )
    log(f"训练集大小: {len(loader.dataset)}  steps/epoch: {len(loader)}")

    # 从首个样本推断 n_sp
    sample = loader.dataset
    if hasattr(sample, "datasets"):
        sample = sample.datasets[0]
    sample = sample[0]
    s = sample["od"].shape[0]
    n_sp = s // args.spectral_patch_size
    if s % args.spectral_patch_size != 0:
        raise RuntimeError(f"S={s} 不能整除 s_p={args.spectral_patch_size}")

    model = NMFPretrainModel(
        embed_dim=args.embed_dim,
        spectral_patch_size=args.spectral_patch_size,
        num_endmembers=args.num_endmembers,
        num_spectral_groups=n_sp,
        patch_size=args.patch_size,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        aggregate_mode=args.aggregate_mode,
        abundance_activation=args.abundance_act,
        use_refine=args.use_refine,
        od_max=args.od_max,
    ).cuda()
    # use_cons_token=False 时 token_cons_head 参数不参与梯度，需告知 DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=not args.use_cons_token)
    log(
        f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  n_sp={n_sp}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        args.epochs,
        args.warmup_epochs,
        len(loader),
        args.lr,
        args.min_lr,
    )
    criterion = NMFPretextLoss(
        lambda_od=args.lambda_od,
        lambda_i=args.lambda_i,
        lambda_cons_pix=args.lambda_cons_pix,
        lambda_cons_token=args.lambda_cons_token,
        lambda_anchor=args.lambda_anchor,
        use_cons_token=args.use_cons_token,
        num_endmembers=args.num_endmembers,
        proj_dim=args.proj_dim,
    ).cuda()
    scaler = GradScaler(enabled=args.amp)

    log("开始预训练")
    loss_plotter = PretrainLossPlotter(args.save_dir) if is_main() else None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        summary = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            epoch,
            args,
            sampler=sampler,
        )
        if is_main():
            lr_now = optimizer.param_groups[0]["lr"]
            log(
                f"Epoch {epoch:4d}/{args.epochs}  loss={summary['loss']:.4f}  "
                f"od={summary['loss_od']:.4f}  i={summary['loss_i']:.4f}  "
                f"cons_pix={summary['loss_cons_pix']:.4f}  "
                f"cons_tok={summary['loss_cons_token']:.4f}  "
                f"anchor={summary['loss_anchor']:.4f}  "
                f"skip={summary['skipped_nonfinite']}  "
                f"lr={lr_now:.2e}  time={time.time()-t0:.1f}s"
            )
            loss_plotter.record(epoch, summary)
            loss_plotter.save()
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                save_checkpoint(model, epoch, args.save_dir)

    dist.destroy_process_group()
    log("预训练完成")


if __name__ == "__main__":
    main()
