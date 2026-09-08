"""Fine-tune conditioned HSI backbone with RetinaNet or FCOS detection."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
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

from models.finetune_model_conditioned_detection import ConditionedDetectionModel
from utils.datasets.conditioned_detection_dataset import build_conditioned_detection_loaders
from utils.datasets.detection_view_geometry import DetectionViewConfig
from utils.detection_cli import (
    add_conditioned_model_arguments,
    add_detection_arguments,
    add_detection_view_arguments,
    add_nmf_data_arguments,
    dataset_kwargs_from_args,
    detection_view_config_from_args,
    detection_config_from_args,
    model_config_from_args,
)
from utils.detection_postprocess import DetectionPostProcessor
from utils.detection_runtime import (
    distributed_context,
    evaluate_detection_model,
    move_model_inputs,
)
from utils.detection_reporting import candidate_statistics
from utils.finetune_curve_monitor import (
    DETECTION_CURVE_GROUPS,
    FinetuneCurveMonitor,
)
from utils.losses import DetectionCriterion
from utils.scheduler import build_cosine_scheduler


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditioned HSI detection fine-tuning")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-annotation", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--val-annotation", required=True)
    parser.add_argument("--test-root")
    parser.add_argument("--test-annotation")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--pretrain-ckpt")
    parser.add_argument("--resume")
    parser.add_argument("--freeze-backbone", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation-probability", type=float, default=0.5)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="records/finetune_conditioned_detection/run")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=1,
        help="Run full validation every N epochs (first/final epochs are always evaluated).",
    )
    parser.add_argument(
        "--pr-curve-interval",
        type=int,
        default=5,
        help="Save a validation COCO PR curve every N epochs; <=0 disables it.",
    )
    parser.add_argument(
        "--progress", choices=("tqdm", "log", "none"), default="log"
    )
    parser.add_argument("--log-interval", type=int, default=10)
    add_conditioned_model_arguments(parser)
    add_detection_arguments(parser)
    add_detection_view_arguments(parser)
    add_nmf_data_arguments(parser)
    args = parser.parse_args()
    if bool(args.test_root) != bool(args.test_annotation):
        parser.error("--test-root and --test-annotation must be supplied together")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers non-negative")
    if args.backbone_lr_mult <= 0:
        parser.error("--backbone-lr-mult must be positive")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.pr_curve_interval < 0:
        parser.error("--pr-curve-interval must be non-negative")
    if args.evaluation_interval < 1:
        parser.error("--evaluation-interval must be positive")
    if not 0 <= args.augmentation_probability <= 1:
        parser.error("--augmentation-probability must be in [0,1]")
    return args


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _checkpoint_payload(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_ap: float,
    model_config,
    detection_config,
    dataset,
    args,
) -> dict:
    return {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_ap50_95": best_ap,
        "model_config": asdict(model_config),
        "detection_config": detection_config.to_dict(),
        "view_config": dataset.view_config.to_dict(),
        "category_id_to_label": dataset.category_id_to_label,
        "category_names": dataset.category_names,
        "args": vars(args),
    }


def _reduce_epoch_totals(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def main() -> None:
    args = get_args()
    distributed, rank, world_size, local_rank = distributed_context()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if args.amp and device.type != "cuda" and rank == 0:
        print("[Detection] --amp ignored because CUDA is unavailable", flush=True)
    amp_enabled = args.amp and device.type == "cuda"

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    view_config = detection_view_config_from_args(args)
    if view_config.model_input_size is not None:
        if (
            view_config.model_input_size[0] % args.patch_size
            or view_config.model_input_size[1] % args.patch_size
        ):
            raise ValueError(
                "--model-input-size must be divisible by --patch-size on both axes"
            )
    data_kwargs = dataset_kwargs_from_args(args, view_config)
    data_kwargs["augmentation_probability"] = args.augmentation_probability
    train_loader, val_loader, test_loader, train_sampler = build_conditioned_detection_loaders(
        args.train_root,
        args.train_annotation,
        args.val_root,
        args.val_annotation,
        args.test_root,
        args.test_annotation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        augment=args.augment,
        **data_kwargs,
    )
    inferred_classes = train_loader.dataset.num_classes
    if args.num_classes is not None and args.num_classes != inferred_classes:
        raise ValueError(
            f"--num-classes={args.num_classes}, dataset categories imply {inferred_classes}"
        )
    model_config = model_config_from_args(args)
    detection_config = detection_config_from_args(args, inferred_classes)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_model_config = resume_state.get("model_config")
        saved_detection_config = resume_state.get("detection_config")
        if saved_model_config != asdict(model_config):
            raise ValueError("resume checkpoint model_config differs from current arguments")
        if saved_detection_config != detection_config.to_dict():
            raise ValueError("resume checkpoint detection_config differs from current arguments")
        saved_view_config = DetectionViewConfig.from_dict(
            resume_state.get("view_config", {"view_mode": "direct"})
        ).to_dict()
        if saved_view_config != view_config.to_dict():
            raise ValueError("resume checkpoint view_config differs from current arguments")
        saved_mapping = {
            int(key): int(value)
            for key, value in resume_state.get("category_id_to_label", {}).items()
        }
        if saved_mapping != train_loader.dataset.category_id_to_label:
            raise ValueError("resume checkpoint category mapping differs from training dataset")
    model = ConditionedDetectionModel(
        model_config,
        detection_config,
        args.pretrain_ckpt if not args.resume else None,
        args.freeze_backbone,
    ).to(device)
    if distributed and device.type == "cuda" and not args.freeze_backbone:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=not args.freeze_backbone,
        )

    inner = _unwrap(model)
    backbone_parameters = [
        parameter for parameter in inner.backbone.parameters() if parameter.requires_grad
    ]
    detector_parameters = [
        parameter
        for name, parameter in inner.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    groups = [{"params": detector_parameters, "lr": args.lr, "name": "detector"}]
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": args.lr * args.backbone_lr_mult,
                "name": "backbone",
            }
        )
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    if not len(train_loader):
        raise RuntimeError("empty training loader")
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        args.epochs,
        args.warmup_epochs,
        optimizer_steps_per_epoch,
        args.lr,
        args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = DetectionCriterion(detection_config).to(device)
    postprocessor = DetectionPostProcessor(detection_config)

    start_epoch, best_ap = 1, -1.0
    if resume_state is not None:
        inner.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_ap = float(resume_state.get("best_ap50_95", -1.0))

    if rank == 0:
        resolved = {
            "model_config": asdict(model_config),
            "detection_config": detection_config.to_dict(),
            "view_config": view_config.to_dict(),
            "category_id_to_label": train_loader.dataset.category_id_to_label,
            "args": vars(args),
            "world_size": world_size,
        }
        (save_dir / "config_resolved.json").write_text(
            json.dumps(resolved, indent=2), encoding="utf-8"
        )
        statistics = candidate_statistics(
            inner, detection_config, train_loader.dataset.spatial_shape
        )
        (save_dir / "candidate_statistics.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
        (save_dir / "validation_view_manifest.json").write_text(
            json.dumps(val_loader.dataset.evaluation_view_manifest, indent=2),
            encoding="utf-8",
        )
        if test_loader is not None:
            (save_dir / "test_view_manifest.json").write_text(
                json.dumps(test_loader.dataset.evaluation_view_manifest, indent=2),
                encoding="utf-8",
            )
        print(
            f"[Detection] mode={detection_config.detection_mode} "
            f"features={detection_config.feature_mode} classes={inferred_classes} "
            f"view_mode={view_config.view_mode} "
            f"train_views={len(train_loader.dataset)} val_views={len(val_loader.dataset)}",
            flush=True,
        )
        print(
            f"[Detection] candidates/image={statistics['total_candidates_per_image']} "
            f"parameters={statistics['parameters']} "
            f"trainable={statistics['trainable_parameters']} "
            f"accumulation={args.gradient_accumulation_steps} "
            f"effective_batch={args.batch_size * world_size * args.gradient_accumulation_steps}",
            flush=True,
        )

    history_path = save_dir / "history.jsonl"
    curve_monitor = (
        FinetuneCurveMonitor(save_dir, DETECTION_CURVE_GROUPS)
        if rank == 0
        else None
    )
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        totals = torch.zeros(8, dtype=torch.float64, device=device)
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        iterable = enumerate(train_loader, start=1)
        if args.progress == "tqdm" and rank == 0:
            iterable = tqdm(
                iterable,
                total=len(train_loader),
                desc=f"Det {epoch:04d}",
                leave=False,
            )
        for step, (model_inputs, targets) in iterable:
            model_inputs = move_model_inputs(model_inputs, device)
            group_start = ((step - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps + 1
            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start + 1,
            )
            should_step = (
                step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            synchronization = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else contextlib.nullcontext()
            )
            with synchronization:
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    raw = model(model_inputs)
                    losses = criterion(raw, targets)
                    backward_loss = losses["loss_total"] / group_size
                scaler.scale(backward_loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            batch_size = len(targets)
            totals[0] += float(losses["loss_total"].detach()) * batch_size
            totals[1] += float(losses["loss_cls"].detach()) * batch_size
            totals[2] += float(losses["loss_box"].detach()) * batch_size
            totals[3] += float(losses.get("loss_centerness", torch.tensor(0.0)).detach()) * batch_size
            totals[4] += float(losses["num_positive"].detach())
            totals[5] += float(losses["num_negative"].detach())
            totals[6] += float(losses["num_ignored"].detach())
            totals[7] += batch_size
            if (
                args.progress == "log"
                and rank == 0
                and args.log_interval > 0
                and step % args.log_interval == 0
            ):
                print(
                    f"[Train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={float(losses['loss_total']):.5f} "
                    f"pos={int(losses['num_positive'])} neg={int(losses['num_negative'])} "
                    f"ignore={int(losses['num_ignored'])}",
                    flush=True,
                )
        totals = _reduce_epoch_totals(totals, distributed)
        train_metrics = {
            "loss_total": float(totals[0] / totals[7].clamp(min=1)),
            "loss_cls": float(totals[1] / totals[7].clamp(min=1)),
            "loss_box": float(totals[2] / totals[7].clamp(min=1)),
            "loss_centerness": float(totals[3] / totals[7].clamp(min=1)),
            "num_positive": int(totals[4]),
            "num_negative": int(totals[5]),
            "num_ignored": int(totals[6]),
        }
        perform_evaluation = (
            epoch == start_epoch
            or epoch == args.epochs
            or epoch % args.evaluation_interval == 0
        )
        val_metrics = None
        improved = False
        if perform_evaluation:
            save_pr_curve = (
                args.pr_curve_interval > 0 and epoch % args.pr_curve_interval == 0
            )
            val_metrics = evaluate_detection_model(
                model,
                val_loader,
                postprocessor,
                device,
                amp=amp_enabled,
                distributed=distributed,
                rank=rank,
                output_dir=save_dir / "validation_latest" if rank == 0 else None,
                pr_curve_path=(
                    save_dir / "pr_curves" / f"validation_pr_epoch{epoch:04d}.png"
                    if save_pr_curve
                    else None
                ),
            )
            current_ap = float(val_metrics["AP50_95"])
            improved = current_ap > best_ap or best_ap < 0
            if improved:
                best_ap = current_ap
                no_improve = 0
            else:
                no_improve += 1
        if rank == 0:
            payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_ap,
                model_config,
                detection_config,
                train_loader.dataset,
                args,
            )
            torch.save(payload, save_dir / "ckpt_last.pth")
            if improved:
                torch.save(payload, save_dir / "ckpt_best.pth")
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                torch.save(payload, save_dir / f"ckpt_epoch{epoch:04d}.pth")
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "elapsed_seconds": time.time() - epoch_start,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            learning_rates = {
                str(group.get("name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            assert curve_monitor is not None
            curve_values = {
                "train_loss_total": train_metrics["loss_total"],
                "train_loss_cls": train_metrics["loss_cls"],
                "train_loss_box": train_metrics["loss_box"],
                "train_loss_centerness": train_metrics["loss_centerness"],
                "num_positive": train_metrics["num_positive"],
                "num_negative": train_metrics["num_negative"],
                "num_ignored": train_metrics["num_ignored"],
                "detector_lr": learning_rates["detector"],
            }
            if val_metrics is not None:
                curve_values.update(
                    val_ap50_95=val_metrics["AP50_95"],
                    val_ap50=val_metrics["AP50"],
                    val_ap75=val_metrics["AP75"],
                    val_ar100=val_metrics["AR_100"],
                )
            if "backbone" in learning_rates:
                curve_values["backbone_lr"] = learning_rates["backbone"]
            curve_monitor.record(epoch, curve_values)
            if val_metrics is not None:
                print(
                    f"[Epoch {epoch}] train_loss={train_metrics['loss_total']:.5f} "
                    f"val_AP50:95={float(val_metrics['AP50_95']):.4f} "
                    f"val_AP50={float(val_metrics['AP50']):.4f} best={best_ap:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[Epoch {epoch}] train_loss={train_metrics['loss_total']:.5f} "
                    f"validation=skipped best={best_ap:.4f}",
                    flush=True,
                )
        stop = bool(args.early_stop and no_improve >= args.patience)
        stop_tensor = torch.tensor(int(stop), device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            break

    if distributed:
        dist.barrier()
    if test_loader is not None:
        best_state = torch.load(save_dir / "ckpt_best.pth", map_location="cpu", weights_only=False)
        _unwrap(model).load_state_dict(best_state["model"])
        test_metrics = evaluate_detection_model(
            model,
            test_loader,
            postprocessor,
            device,
            amp=amp_enabled,
            distributed=distributed,
            rank=rank,
            output_dir=save_dir / "test_best" if rank == 0 else None,
        )
        if rank == 0:
            print(
                f"[Test best] AP50:95={test_metrics['AP50_95']:.4f} "
                f"AP50={test_metrics['AP50']:.4f} AP75={test_metrics['AP75']:.4f}",
                flush=True,
            )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
