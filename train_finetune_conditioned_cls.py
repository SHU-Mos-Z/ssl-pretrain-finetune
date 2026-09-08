"""Distributed patch-classification fine-tuning for the conditioned backbone."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.conditioned_contracts import ConditionedModelConfig
from models.finetune_model_conditioned_cls import (
    CLASSIFICATION_HEAD_TYPES,
    ConditionedClassificationModel,
)
from utils.classification_metrics import compute_classification_metrics
from utils.datasets.conditioned_classification_dataset import (
    CLASSIFICATION_AUGMENTATION_POLICIES,
    build_conditioned_classification_loaders,
)
from utils.augmentations.hsi_spatial import PERSPECTIVE_PADDING_MODES
from utils.finetune_curve_monitor import (
    CLASSIFICATION_CURVE_GROUPS,
    FinetuneCurveMonitor,
)
from utils.scheduler import build_cosine_scheduler


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditioned backbone patch-classification fine-tuning"
    )
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root")
    parser.add_argument("--class-map-file")
    parser.add_argument(
        "--train-exclude-json",
        help=(
            "Optional top-level JSON list containing class_name/stem pairs to exclude "
            "from the training split only. Validation and test splits are unchanged."
        ),
    )
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--wavelength-file")
    parser.add_argument("--allow-index-wavelengths", action="store_true")

    parser.add_argument("--nmf-k", type=int, default=16)
    parser.add_argument("--nmf-l1", type=float, default=5e-4)
    parser.add_argument("--nmf-l2", type=float, default=2e-4)
    parser.add_argument("--nmf-l3", type=float, default=1e-2)
    parser.add_argument(
        "--nmf-simplex", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--nmf-lam-e", type=float, default=0.05)
    parser.add_argument("--nmf-e-clamp-max", type=float, default=3.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weighting", choices=("none", "inverse_freq"), default="none"
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("standard", "balanced"),
        default="standard",
        help=(
            "Training sampler. 'balanced' samples classes with inverse-frequency "
            "replacement and shards the deterministic draw across distributed ranks."
        ),
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help=(
            "Number of virtual online views per original training sample and epoch. "
            "Dihedral-enabled policies use distinct transforms for copies of the "
            "same sample; must be in [1,8]. Validation/test data are never expanded."
        ),
    )
    parser.add_argument(
        "--augmentation-policy",
        choices=CLASSIFICATION_AUGMENTATION_POLICIES,
        default="dihedral",
        help=(
            "Spatial augmentation family. 'dihedral' preserves the historical "
            "eight rotations/reflections; 'perspective' uses only four-point "
            "perspective; 'dihedral_perspective' composes both."
        ),
    )
    parser.add_argument(
        "--perspective-probability",
        type=float,
        default=0.5,
        help="Per-view probability of perspective when the selected policy enables it.",
    )
    parser.add_argument(
        "--perspective-scale",
        type=float,
        default=0.05,
        help="Maximum inward corner displacement relative to full image size.",
    )
    parser.add_argument(
        "--perspective-padding-mode",
        choices=PERSPECTIVE_PADDING_MODES,
        default="reflection",
    )
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--pretrain-ckpt")
    parser.add_argument("--resume")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--classification-head",
        choices=CLASSIFICATION_HEAD_TYPES,
        default="h0_gap_linear",
        help="Classification head used for the H0-H3 ablation.",
    )
    parser.add_argument("--head-projection-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--spectral-patch-size", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-stem-ch", type=int, default=64)
    parser.add_argument(
        "--cnn-spectral-agg",
        choices=("mean", "max", "attention"),
        default="attention",
    )
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--decoder-mid-ch", type=int, default=64)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--confidence-temperature", type=float, default=0.05)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-extra", type=float, default=1.0)
    parser.add_argument("--od-max", type=float, default=3.0)

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="records/finetune_conditioned_cls/run")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--best-val-interval",
        type=int,
        default=10,
        help=(
            "按此 epoch 数划窗口，记录每个窗口内验证集 Macro-F1 最佳的模型 "
            "（ckpt_window{start}-{end}_best.pth）；训练结束后逐个在测试集上评估，"
            "避免验证集全局最优过早出现导致后续 epoch 被完全忽视。<=0 时禁用窗口机制。"
        ),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--progress", choices=("tqdm", "log", "none"), default="log"
    )
    args = parser.parse_args()
    if args.backbone_lr_mult <= 0:
        parser.error("--backbone-lr-mult must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0,1)")
    if not 1 <= args.augmentation_copies <= 8:
        parser.error("--augmentation-copies must be in [1,8]")
    if not args.augment and args.augmentation_copies != 1:
        parser.error("--augmentation-copies must be 1 when --no-augment is used")
    if not 0.0 <= args.perspective_probability <= 1.0:
        parser.error("--perspective-probability must be in [0,1]")
    if not 0.0 <= args.perspective_scale < 0.5:
        parser.error("--perspective-scale must be in [0,0.5)")
    if args.head_projection_dim <= 0 or args.head_hidden_dim <= 0:
        parser.error("classification head dimensions must be positive")
    return args


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _gather_predictions(
    local_logits: list[torch.Tensor],
    local_targets: list[torch.Tensor],
    local_indices: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    payload = {
        "logits": torch.cat(local_logits).float().numpy()
        if local_logits
        else np.empty((0, 0), dtype=np.float32),
        "targets": torch.cat(local_targets).long().numpy()
        if local_targets
        else np.empty((0,), dtype=np.int64),
        "indices": torch.cat(local_indices).long().numpy()
        if local_indices
        else np.empty((0,), dtype=np.int64),
    }
    gathered: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() != 0:
        return None
    valid = [item for item in gathered if item is not None and item["indices"].size]
    logits = np.concatenate([item["logits"] for item in valid], axis=0)
    targets = np.concatenate([item["targets"] for item in valid], axis=0)
    indices = np.concatenate([item["indices"] for item in valid], axis=0)
    order = np.argsort(indices, kind="stable")
    indices, logits, targets = indices[order], logits[order], targets[order]
    if np.unique(indices).size != indices.size:
        raise RuntimeError("evaluation sampler produced duplicate sample indices")
    return logits, targets, indices


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    model.eval()
    local_logits: list[torch.Tensor] = []
    local_targets: list[torch.Tensor] = []
    local_indices: list[torch.Tensor] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
        local_logits.append(logits.detach().float().cpu())
        local_targets.append(batch["label"].detach().cpu())
        local_indices.append(batch["sample_index"].detach().cpu())

    collected = _gather_predictions(local_logits, local_targets, local_indices)
    result: list[dict | None] = [None]
    if dist.get_rank() == 0:
        assert collected is not None
        logits_np, targets_np, _ = collected
        result[0] = compute_classification_metrics(
            logits_np, targets_np, num_classes
        )
    dist.broadcast_object_list(result, src=0)
    assert result[0] is not None
    return result[0], collected


def _class_weights(counts: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = counts.float()
    if torch.any(counts <= 0):
        raise ValueError("every class must have at least one training sample")
    weights = counts.sum() / (counts.numel() * counts)
    return weights / weights.mean()


def _state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return (model.module if isinstance(model, DDP) else model).state_dict()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_macro_f1: float,
    config: ConditionedModelConfig,
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": _state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_macro_f1": best_macro_f1,
            "model_config": asdict(config),
            "class_to_idx": class_to_idx,
            "args": vars(args),
        },
        path,
    )


def load_resume(path: str, model: nn.Module, optimizer, scheduler) -> tuple[int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    inner = model.module if isinstance(model, DDP) else model
    inner.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]) + 1, float(state.get("best_macro_f1", 0.0))


def _write_predictions(
    path: Path,
    collected: tuple[np.ndarray, np.ndarray, np.ndarray],
    stems: list[str],
    class_to_idx: dict[str, int],
) -> None:
    logits, targets, indices = collected
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "stem", "target", "prediction"] + [f"p_{name}" for name in names])
        for row, sample_index in enumerate(indices):
            writer.writerow(
                [
                    int(sample_index),
                    stems[int(sample_index)],
                    int(targets[row]),
                    int(predictions[row]),
                    *[float(value) for value in probabilities[row]],
                ]
            )


def main() -> None:
    args = get_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_kwargs = dict(
        class_map_file=args.class_map_file,
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        nmf_k=args.nmf_k,
        nmf_l1=args.nmf_l1,
        nmf_l2=args.nmf_l2,
        nmf_l3=args.nmf_l3,
        nmf_simplex=args.nmf_simplex,
        nmf_lam_e=args.nmf_lam_e,
        nmf_e_clamp_max=args.nmf_e_clamp_max,
        wavelength_file=args.wavelength_file,
        allow_index_wavelengths=args.allow_index_wavelengths,
        od_max=args.od_max,
        augment=args.augment,
        augmentation_copies=args.augmentation_copies,
        augmentation_seed=args.seed,
        augmentation_policy=args.augmentation_policy,
        perspective_probability=args.perspective_probability,
        perspective_scale=args.perspective_scale,
        perspective_padding_mode=args.perspective_padding_mode,
    )
    train_loader, val_loader, test_loader, train_sampler, class_to_idx = (
        build_conditioned_classification_loaders(
            args.train_root,
            args.val_root,
            args.test_root,
            args.batch_size,
            args.workers,
            True,
            rank,
            world_size,
            sampling_strategy=args.sampling_strategy,
            sampling_seed=args.seed,
            train_exclude_samples_file=args.train_exclude_json,
            **data_kwargs,
        )
    )
    inferred_classes = len(class_to_idx)
    if args.num_classes is not None and args.num_classes != inferred_classes:
        raise ValueError(
            f"--num-classes={args.num_classes}, but the dataset has "
            f"{inferred_classes} classes: {class_to_idx}"
        )
    num_classes = inferred_classes
    if rank == 0:
        (save_dir / "class_to_idx.json").write_text(
            json.dumps(class_to_idx, indent=2), encoding="utf-8"
        )
        print(f"[Classification] class_to_idx={class_to_idx}", flush=True)
        print(
            "[Classification] augmentation="
            f"enabled={args.augment} copies={args.augmentation_copies} "
            f"policy={args.augmentation_policy} "
            f"perspective_probability={args.perspective_probability} "
            f"perspective_scale={args.perspective_scale} "
            f"padding={args.perspective_padding_mode}",
            flush=True,
        )
        exclusion_summary = train_loader.dataset.exclusion_summary
        (save_dir / "training_sample_exclusion.json").write_text(
            json.dumps(exclusion_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[Classification] training sample exclusion="
            f"{exclusion_summary}",
            flush=True,
        )

    config = ConditionedModelConfig(
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
    model = ConditionedClassificationModel(
        num_classes,
        config,
        args.pretrain_ckpt,
        args.freeze_backbone,
        args.head_dropout,
        args.classification_head,
        args.head_projection_dim,
        args.head_hidden_dim,
    ).to(device)
    if not args.freeze_backbone:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        broadcast_buffers=not args.freeze_backbone,
    )

    inner = model.module
    if rank == 0:
        head_parameter_count = sum(
            parameter.numel() for parameter in inner.cls_head.parameters()
        )
        print(
            f"[Classification] head={args.classification_head} "
            f"head_parameters={head_parameter_count:,} "
            f"projection_dim={args.head_projection_dim} "
            f"hidden_dim={args.head_hidden_dim}",
            flush=True,
        )
    head_parameters = [
        parameter
        for parameter in inner.cls_head.parameters()
        if parameter.requires_grad
    ]
    backbone_parameters = [
        parameter for parameter in inner.backbone.parameters() if parameter.requires_grad
    ]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_mult}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    if len(train_loader) == 0:
        raise ValueError(
            "training loader is empty; reduce --batch-size/--world-size or add samples"
        )
    scheduler = build_cosine_scheduler(
        optimizer,
        args.epochs,
        args.warmup_epochs,
        len(train_loader),
        args.lr,
        args.min_lr,
    )
    weights = _class_weights(train_loader.dataset.class_counts(), args.class_weighting)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )

    start_epoch, best_macro_f1 = 1, -1.0
    if args.resume:
        start_epoch, best_macro_f1 = load_resume(
            args.resume, model, optimizer, scheduler
        )
    best_path = save_dir / "ckpt_best.pth"
    last_path = save_dir / "ckpt_last.pth"
    curve_monitor = (
        FinetuneCurveMonitor(save_dir, CLASSIFICATION_CURVE_GROUPS)
        if rank == 0
        else None
    )
    no_improve = 0
    # 窗口最佳：window_key -> 该窗口内已见过的最高 Val Macro-F1（仅 rank0 使用）
    window_best_f1: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        assert train_sampler is not None
        train_sampler.set_epoch(epoch)
        totals = torch.zeros(3, device=device)
        start_time = time.time()
        iterable = enumerate(train_loader)
        if args.progress == "tqdm" and rank == 0:
            iterable = tqdm(
                iterable,
                total=len(train_loader),
                desc=f"Cls {epoch:04d}",
                leave=False,
            )
        for step, batch in iterable:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits = model(batch)
                loss = criterion(logits, batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite classification loss at epoch {epoch}")
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            batch_size = batch["label"].numel()
            totals[0] += loss.detach() * batch_size
            totals[1] += (logits.argmax(dim=1) == batch["label"]).sum()
            totals[2] += batch_size
            if (
                args.progress == "log"
                and rank == 0
                and args.log_interval > 0
                and (step + 1) % args.log_interval == 0
            ):
                print(
                    f"[{time.strftime('%F %T')}] epoch={epoch} "
                    f"step={step+1}/{len(train_loader)} loss={loss.item():.5f}",
                    flush=True,
                )
        dist.all_reduce(totals)
        val_metrics, _ = evaluate(model, val_loader, num_classes, device, args.amp)
        current_f1 = float(val_metrics["MacroF1"])
        stop = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0:
            train_loss = float(totals[0] / totals[2].clamp(min=1))
            train_accuracy = float(totals[1] / totals[2].clamp(min=1))
            record = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_metrics["Accuracy"],
                "val_macro_f1": val_metrics["MacroF1"],
                "val_macro_auc": val_metrics["MacroAUC"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            assert curve_monitor is not None
            curve_monitor.record(epoch, record)
            print(
                f"[{time.strftime('%F %T')}] epoch={epoch}/{args.epochs} "
                f"loss={train_loss:.5f} train_acc={train_accuracy:.4f} "
                f"val_acc={val_metrics['Accuracy']:.4f} "
                f"val_f1={val_metrics['MacroF1']:.4f} "
                f"val_auc={val_metrics['MacroAUC']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"time={time.time()-start_time:.1f}s",
                flush=True,
            )
            improved = current_f1 > best_macro_f1
            if improved:
                best_macro_f1 = current_f1
                no_improve = 0
            else:
                no_improve += 1
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_f1,
                config,
                class_to_idx,
                args,
            )
            if improved:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_macro_f1,
                    config,
                    class_to_idx,
                    args,
                )
                print(f">>> New best Macro-F1: {best_macro_f1:.4f}", flush=True)
            if epoch % args.save_interval == 0:
                save_checkpoint(
                    save_dir / f"ckpt_epoch{epoch:04d}.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_macro_f1,
                    config,
                    class_to_idx,
                    args,
                )
            if args.best_val_interval > 0:
                window_idx = (epoch - 1) // args.best_val_interval
                window_start = window_idx * args.best_val_interval + 1
                window_end = window_start + args.best_val_interval - 1
                window_key = f"{window_start:04d}-{window_end:04d}"
                if current_f1 > window_best_f1.get(window_key, -1.0):
                    window_best_f1[window_key] = current_f1
                    save_checkpoint(
                        save_dir / f"ckpt_window{window_key}_best.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_macro_f1,
                        config,
                        class_to_idx,
                        args,
                    )
                    print(
                        f">>> [Window {window_key}] New best Macro-F1: "
                        f"{current_f1:.4f} at epoch {epoch}",
                        flush=True,
                    )
            if args.early_stop and no_improve >= args.patience:
                stop.fill_(1)
        dist.broadcast(stop, src=0)
        if stop.item():
            break

    if test_loader is not None:
        dist.barrier()

        def _load_and_test(ckpt_path) -> tuple[dict, tuple | None]:
            checkpoint: list[dict | None] = [
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if rank == 0
                else None
            ]
            dist.broadcast_object_list(checkpoint, src=0)
            assert checkpoint[0] is not None
            model.module.load_state_dict(checkpoint[0]["model"])
            return evaluate(model, test_loader, num_classes, device, args.amp)

        # 先逐个测试每个窗口内验证集最佳的模型，避免全局最优过早出现时，
        # 后续 epoch 学到的信息被完全忽视而无法在测试集上体现。
        if args.best_val_interval > 0:
            if rank == 0:
                window_paths = sorted(
                    glob.glob(str(save_dir / "ckpt_window*_best.pth"))
                )
            else:
                window_paths = None
            obj = [window_paths]
            dist.broadcast_object_list(obj, src=0)
            window_paths = obj[0]
            for window_path in window_paths:
                label = Path(window_path).stem
                window_metrics, _ = _load_and_test(window_path)
                if rank == 0:
                    (save_dir / f"test_metrics_{label}.json").write_text(
                        json.dumps(window_metrics, indent=2), encoding="utf-8"
                    )
                    print(
                        f"[{label}] Test Accuracy: {window_metrics['Accuracy']:.4f}  "
                        f"Macro-F1: {window_metrics['MacroF1']:.4f}  "
                        f"Macro-AUC: {window_metrics['MacroAUC']:.4f}",
                        flush=True,
                    )

        test_metrics, collected = _load_and_test(best_path)
        if rank == 0:
            (save_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, indent=2), encoding="utf-8"
            )
            assert collected is not None
            _write_predictions(
                save_dir / "test_predictions.csv",
                collected,
                [sample.stem for sample in test_loader.dataset.samples],
                class_to_idx,
            )
            print(
                f"Test Accuracy: {test_metrics['Accuracy']:.4f}  "
                f"Macro-F1: {test_metrics['MacroF1']:.4f}  "
                f"Macro-AUC: {test_metrics['MacroAUC']:.4f}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
