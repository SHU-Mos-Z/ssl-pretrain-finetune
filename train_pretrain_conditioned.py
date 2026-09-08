"""Distributed endmember-conditioned pretraining entry point."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.conditioned_contracts import ConditionedModelConfig
from models.endmember_conditioned_pretrain_model import (
    EndmemberConditionedPretrainModel,
)
from utils.conditioned_checkpoint import (
    resume_training_checkpoint,
    save_training_checkpoint,
)
from utils.conditioned_pretrain_monitor import ConditionedPretrainMonitor
from utils.datasets import (
    build_conditioned_pretrain_loader,
    set_conditioned_dataset_epoch,
)
from utils.losses import ConditionedPretextLoss
from utils.scheduler import build_cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(
        description="Endmember-conditioned HSI pathology pretraining"
    )
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--allow-index-wavelengths", action="store_true")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--spectral-patch-size", type=int, default=5)
    p.add_argument("--spectral-mask-ratio", type=float, default=0.3)
    p.add_argument("--spatial-mask-ratio", type=float, default=0.2)
    p.add_argument(
        "--pad-to-patch", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--permute-endmembers", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--nmf-k", type=int, default=16)
    p.add_argument("--nmf-l1", type=float, default=5e-4)
    p.add_argument("--nmf-l2", type=float, default=2e-4)
    p.add_argument("--nmf-l3", type=float, default=1e-2)
    p.add_argument("--nmf-simplex", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nmf-lam-e", type=float, default=0.05)
    p.add_argument("--nmf-e-clamp-max", type=float, default=3.0)
    p.add_argument("--nmf-weight-temperature", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--vit-depth", type=int, default=6)
    p.add_argument("--vit-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cnn-stem-ch", type=int, default=64)
    p.add_argument(
        "--cnn-spectral-agg", choices=["mean", "max", "attention"], default="attention"
    )
    p.add_argument("--fusion-heads", type=int, default=8)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--decoder-mid-ch", type=int, default=64)
    p.add_argument("--residual-hidden-dim", type=int, default=128)
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument("--confidence-temperature", type=float, default=0.05)
    p.add_argument("--alpha-min", type=float, default=0.1)
    p.add_argument("--alpha-extra", type=float, default=1.0)
    p.add_argument("--od-max", type=float, default=3.0)
    for name, default in (
        ("od", 1.0),
        ("i", 1.0),
        ("c", 0.2),
        ("token", 1.0),
        ("feature", 0.0),
        ("delta", 0.01),
        ("sam", 0.0),
    ):
        p.add_argument(f"--lambda-{name}", type=float, default=default)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="records/pretrain_conditioned/run")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--progress", choices=["tqdm", "log", "none"], default="log")
    p.add_argument("--log-interval", type=int, default=20)
    return p.parse_args()


def main():
    args = get_args()
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(args.save_dir, exist_ok=True)
    second_view = args.lambda_feature > 0
    loader, sampler = build_conditioned_pretrain_loader(
        args.root,
        args.batch_size,
        args.workers,
        True,
        nmf_k=args.nmf_k,
        nmf_l1=args.nmf_l1,
        nmf_l2=args.nmf_l2,
        nmf_l3=args.nmf_l3,
        nmf_simplex=args.nmf_simplex,
        nmf_lam_e=args.nmf_lam_e,
        nmf_e_clamp_max=args.nmf_e_clamp_max,
        allow_index_wavelengths=args.allow_index_wavelengths,
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        spectral_mask_ratio=args.spectral_mask_ratio,
        spatial_mask_ratio=args.spatial_mask_ratio,
        second_view=second_view,
        permute_endmembers=args.permute_endmembers,
        od_max=args.od_max,
        nmf_weight_temperature=args.nmf_weight_temperature,
        pad_to_patch=args.pad_to_patch,
        seed=args.seed,
    )
    cfg = ConditionedModelConfig(
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        embed_dim=args.embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        cnn_stem_ch=args.cnn_stem_ch,
        cnn_spectral_agg=args.cnn_spectral_agg,
        fusion_heads=args.fusion_heads,
        feature_dim=args.feature_dim,
        decoder_mid_ch=args.decoder_mid_ch,
        residual_hidden_dim=args.residual_hidden_dim,
        ridge_lambda=args.ridge_lambda,
        confidence_temperature=args.confidence_temperature,
        alpha_min=args.alpha_min,
        alpha_extra=args.alpha_extra,
        od_max=args.od_max,
    )
    model = DDP(EndmemberConditionedPretrainModel(cfg).cuda(), device_ids=[local_rank])
    criterion = ConditionedPretextLoss(
        **{
            f"lambda_{k}": getattr(args, f"lambda_{k}")
            for k in ("od", "i", "c", "token", "feature", "delta", "sam")
        }
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_cosine_scheduler(
        optimizer, args.epochs, args.warmup_epochs, len(loader), args.lr, args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start = 1
    if args.resume:
        epoch, _ = resume_training_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start = epoch + 1
    monitor = ConditionedPretrainMonitor(args.save_dir) if rank == 0 else None
    keys = [
        "loss_total",
        "loss_od",
        "loss_i",
        "loss_c",
        "loss_token",
        "loss_feature",
        "loss_delta",
        "loss_sam",
    ]
    for epoch in range(start, args.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        set_conditioned_dataset_epoch(loader, epoch)
        # loss sums + processed step count + skipped step count
        sums = torch.zeros(len(keys) + 2, device=local_rank)
        iterable = enumerate(loader)
        if args.progress == "tqdm" and rank == 0:
            iterable = tqdm(
                iterable, total=len(loader), desc=f"Epoch {epoch:04d}", leave=False
            )
        t0 = time.time()
        for step, batch in iterable:
            batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                if second_view:
                    batch_size = batch["od"].shape[0]
                    view_b = dict(batch)
                    view_b["token_visible"] = batch["token_visible_b"]
                    view_b["voxel_visible"] = batch["voxel_visible_b"]
                    merged = {}
                    for key, value in batch.items():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim
                            and value.shape[0] == batch_size
                        ):
                            merged[key] = torch.cat((value, view_b[key]), dim=0)
                        else:
                            merged[key] = value
                    merged_out = model(merged)
                    out = {key: value[:batch_size] for key, value in merged_out.items()}
                    out_b = {
                        key: value[batch_size:] for key, value in merged_out.items()
                    }
                else:
                    out = model(batch)
                    out_b = None
                loss, logs = criterion(out, batch, out_b)
            if not torch.isfinite(loss):
                sums[-1] += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for i, key in enumerate(keys):
                sums[i] += logs[key]
            sums[-2] += 1
            if (
                args.progress == "log"
                and rank == 0
                and args.log_interval > 0
                and (step + 1) % args.log_interval == 0
            ):
                print(
                    f"[{time.strftime('%F %T')}]   step {step+1}/{len(loader)}  "
                    f"loss={logs['loss_total']:.4f}",
                    flush=True,
                )
        dist.all_reduce(sums)
        denom = max(float(sums[-2]), 1.0)
        summary = {key: float(sums[i]) / denom for i, key in enumerate(keys)}
        summary["skipped_nonfinite"] = float(sums[-1])
        if rank == 0:
            print(
                f"[{time.strftime('%F %T')}] epoch={epoch}/{args.epochs} "
                + " ".join(f"{k}={v:.5f}" for k, v in summary.items())
                + f" lr={optimizer.param_groups[0]['lr']:.2e} time={time.time()-t0:.1f}s",
                flush=True,
            )
            monitor.record(epoch, summary)
            save_training_checkpoint(
                os.path.join(args.save_dir, "ckpt_last.pth"),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                cfg,
                args,
            )
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                save_training_checkpoint(
                    os.path.join(args.save_dir, f"ckpt_epoch{epoch:04d}.pth"),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    cfg,
                    args,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
