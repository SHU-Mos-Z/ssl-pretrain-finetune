"""Distributed segmentation fine-tuning for the conditioned backbone."""

from __future__ import annotations

import argparse, glob, json, os, random, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.conditioned_contracts import ConditionedModelConfig
from models.finetune_model_conditioned import ConditionedFinetuneModel
from utils.datasets import build_conditioned_finetune_loaders
from utils.datasets.conditioned_finetune_dataset import (
    ConditionedSlidingWindowSceneDataset,
    ConditionedSlidingWindowTestDataset,
    collate_conditioned_model_inputs,
)
from utils.finetune_curve_monitor import (
    FinetuneCurveMonitor,
    build_segmentation_curve_groups,
)
from utils.losses import build_segmentation_criterion, primary_segmentation_logits
from utils.metrics import (
    DICE_BATCH_ALLCLASS_MACRO,
    DICE_CLASSWISE,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_FG,
    IOU_METRIC_NAMES,
    SegmentationMetricAccumulator,
    normalize_dice_metric_names,
)
from utils.scheduler import build_cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(
        description="Conditioned backbone segmentation fine-tuning"
    )
    p.add_argument("--train-root", required=True)
    p.add_argument("--val-root", required=True)
    p.add_argument(
        "--scene-val-root",
        default=None,
        help=(
            "Optional directory of complete validation scenes. When set, sliding-window "
            "scene validation is used for best/window checkpoint selection while "
            "--val-root remains the inexpensive patch-validation source."
        ),
    )
    p.add_argument("--test-root")
    p.add_argument("--wavelength-file", default=None)
    p.add_argument("--allow-index-wavelengths", action="store_true")
    p.add_argument("--nmf-k", type=int, default=16)
    p.add_argument("--nmf-l1", type=float, default=5e-4)
    p.add_argument("--nmf-l2", type=float, default=2e-4)
    p.add_argument("--nmf-l3", type=float, default=1e-2)
    p.add_argument("--nmf-simplex", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nmf-lam-e", type=float, default=0.05)
    p.add_argument("--nmf-e-clamp-max", type=float, default=3.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--pretrain-ckpt")
    p.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Skip training and evaluate this complete fine-tuned state dict.",
    )
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0)
    p.add_argument("--backbone-lr-multiplier", type=float, default=1.0)
    p.add_argument("--head-lr-multiplier", type=float, default=1.0)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--spectral-patch-size", type=int, default=5)
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
    p.add_argument(
        "--segmentation-head",
        choices=["h0_simple", "h1_residual", "h2_aspp", "h3_multiscale_aux"],
        default="h0_simple",
    )
    p.add_argument("--head-hidden-channels", type=int, default=128)
    p.add_argument("--head-projection-channels", type=int, default=64)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--aspp-rates", default="1,6,12,18")
    p.add_argument(
        "--segmentation-loss",
        choices=[
            "ce_dice",
            "weighted_ce_dice",
            "focal_dice",
            "weighted_ce_dice_boundary",
        ],
        default="ce_dice",
    )
    p.add_argument("--ce-loss-weight", type=float, default=1.0)
    p.add_argument("--dice-loss-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--boundary-loss-weight", type=float, default=0.0)
    p.add_argument("--aux-loss-weight", type=float, default=0.0)
    p.add_argument(
        "--class-weight-mode",
        choices=["none", "manual", "inverse_sqrt", "inverse_frequency"],
        default="none",
    )
    p.add_argument("--class-weights", default="")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--augmentation-copies", type=int, default=1)
    p.add_argument(
        "--augmentation-policy",
        choices=["dihedral", "dihedral_affine", "dihedral_perspective"],
        default="dihedral",
    )
    p.add_argument("--augmentation-probability", type=float, default=1.0)
    p.add_argument("--affine-rotation-degrees", type=float, default=15.0)
    p.add_argument("--affine-scale-delta", type=float, default=0.1)
    p.add_argument("--affine-translate-fraction", type=float, default=0.05)
    p.add_argument("--perspective-scale", type=float, default=0.05)
    p.add_argument(
        "--augmentation-padding-mode",
        choices=["zeros", "border", "reflection"],
        default="reflection",
    )
    p.add_argument(
        "--endmember-scope", choices=["patch", "scene"], default="patch"
    )
    p.add_argument("--scene-endmember-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between epochs (effective when workers > 0).",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each DataLoader worker.",
    )
    p.add_argument(
        "--distributed-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate each original validation batch on exactly one DDP rank. "
            "Original batch boundaries are preserved for metric compatibility."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="records/finetune_conditioned/run")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument(
        "--best-val-interval",
        type=int,
        default=10,
        help=(
            "按此 epoch 数划窗口，记录每个窗口内验证集 Dice 最佳的模型 "
            "（ckpt_window{start}-{end}_best.pth）；训练结束后逐个在测试集上评估，"
            "避免验证集全局最优过早出现导致后续 epoch 被完全忽视。<=0 时禁用窗口机制。"
        ),
    )
    p.add_argument("--progress", choices=["tqdm", "log", "none"], default="log")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument(
        "--hd95-backend",
        choices=["scipy", "monai"],
        default="scipy",
        help=(
            "HD95 计算后端：scipy=CPU 距离变换（默认，更稳定）；"
            "monai=与 LoTS-Net reference 一致的 GPU 实现"
        ),
    )
    p.add_argument(
        "--dice-metrics",
        default="",
        help=(
            "Comma-separated Dice protocols. Empty input preserves the historical "
            "batch_allclass_macro metric. Unknown names are ignored."
        ),
    )
    p.add_argument(
        "--primary-dice-metric",
        default=DICE_BATCH_ALLCLASS_MACRO,
        help="Scalar Dice protocol used for best checkpoints and early stopping.",
    )
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Debug/smoke-test limit per training epoch; <=0 uses the full loader.",
    )
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Debug/smoke-test limit for each validation/test pass; <=0 uses all batches.",
    )
    p.add_argument(
        "--scene-val-interval",
        type=int,
        default=5,
        help="Run optional complete-scene validation every N epochs; <=0 is invalid when enabled.",
    )
    p.add_argument("--scene-val-window-size", type=int, default=224)
    p.add_argument("--scene-val-window-stride", type=int, default=112)
    p.add_argument("--scene-val-window-batch-size", type=int, default=4)
    p.add_argument(
        "--scene-val-window-blend",
        choices=["uniform", "gaussian"],
        default="gaussian",
    )
    p.add_argument(
        "--test-inference-mode",
        choices=["direct", "sliding_window"],
        default="direct",
        help="direct forwards each stored test image; sliding_window stitches full-scene logits.",
    )
    p.add_argument("--test-window-size", type=int, default=224)
    p.add_argument("--test-window-stride", type=int, default=112)
    p.add_argument("--test-window-batch-size", type=int, default=4)
    p.add_argument(
        "--test-window-blend",
        choices=["uniform", "gaussian"],
        default="gaussian",
        help="Weighting used to blend overlapping window logits.",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model,
    loader,
    num_classes,
    device,
    amp: bool = False,
    hd95_backend: str = "scipy",
    dice_metrics: tuple[str, ...] | list[str] | None = None,
    max_batches: int = 0,
):
    model.eval()
    accumulator = SegmentationMetricAccumulator(
        num_classes=num_classes,
        dice_metrics=dice_metrics,
        hd95_backend=hd95_backend,
    )
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        scene_ids = batch.pop("scene_id", None)
        stems = batch.pop("stem", None)
        if scene_ids is None:
            scene_ids = stems
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        seg = batch.pop("seg")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = primary_segmentation_logits(model(batch))
        accumulator.update(logits, seg, scene_ids=scene_ids)
    accumulator.synchronize_between_processes()
    return accumulator.compute()


def _sliding_window_starts(length: int, window_size: int, stride: int) -> list[int]:
    """Return starts that cover every pixel and always include the far border."""
    padded_length = max(int(length), int(window_size))
    last = padded_length - window_size
    starts = list(range(0, last + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_blend_weight(window_size: int, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones((window_size, window_size), dtype=torch.float32)
    if mode != "gaussian":
        raise ValueError(f"unknown sliding-window blend mode: {mode!r}")
    coordinates = torch.linspace(-1.0, 1.0, window_size, dtype=torch.float32)
    # A strictly positive Gaussian avoids uncovered/zero-weight image borders.
    one_dimensional = torch.exp(-0.5 * (coordinates / 0.5).square()).clamp_min(1e-3)
    return torch.outer(one_dimensional, one_dimensional).clamp_min(1e-3)


@torch.no_grad()
def evaluate_sliding_window(
    model,
    dataset: ConditionedSlidingWindowSceneDataset,
    num_classes: int,
    device,
    *,
    window_size: int = 224,
    stride: int = 112,
    tile_batch_size: int = 4,
    blend: str = "gaussian",
    amp: bool = False,
    hd95_backend: str = "scipy",
    dice_metrics: tuple[str, ...] | list[str] | None = None,
    max_scenes: int = 0,
    progress: str = "log",
    log_interval: int = 10,
    phase_label: str = "Sliding evaluation",
):
    """Predict complete scenes by blending continuous window logits.

    Metrics are updated exactly once per reconstructed scene.  No metric is
    computed on individual or overlapping tiles.
    """
    model.eval()
    accumulator = SegmentationMetricAccumulator(
        num_classes=num_classes,
        dice_metrics=dice_metrics,
        hd95_backend=hd95_backend,
    )
    indices = dataset.distributed_indices(max_scenes=max_scenes)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    iterable = indices
    if progress == "tqdm" and rank == 0:
        iterable = tqdm(indices, desc=phase_label, leave=False)
    blend_weight = _sliding_blend_weight(window_size, blend)

    for local_scene_index, dataset_index in enumerate(iterable):
        scene = dataset.load_scene(dataset_index)
        height, width = int(scene["height"]), int(scene["width"])
        y_starts = _sliding_window_starts(height, window_size, stride)
        x_starts = _sliding_window_starts(width, window_size, stride)
        coordinates = [(y, x) for y in y_starts for x in x_starts]
        logit_sum = torch.zeros((num_classes, height, width), dtype=torch.float32)
        weight_sum = torch.zeros((height, width), dtype=torch.float32)

        for offset in range(0, len(coordinates), tile_batch_size):
            coordinate_batch = coordinates[offset:offset + tile_batch_size]
            tile_samples: list[dict[str, torch.Tensor]] = []
            valid_shapes: list[tuple[int, int]] = []
            for y, x in coordinate_batch:
                valid_height = min(window_size, max(height - y, 0))
                valid_width = min(window_size, max(width - x, 0))
                intensity = dataset.read_tile(
                    scene, y, x, valid_height, valid_width
                )
                if valid_height < window_size or valid_width < window_size:
                    intensity = np.pad(
                        intensity,
                        ((0, 0), (0, window_size - valid_height),
                         (0, window_size - valid_width)),
                        mode="constant",
                        constant_values=1.0,
                    )
                tile_samples.append(
                    dataset.build_tile_inputs(intensity, scene["e_star"])
                )
                valid_shapes.append((valid_height, valid_width))
            model_batch = collate_conditioned_model_inputs(tile_samples)
            model_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in model_batch.items()
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                tile_logits = primary_segmentation_logits(model(model_batch))
            tile_logits = tile_logits.detach().float().cpu()

            for tile_index, ((y, x), (valid_height, valid_width)) in enumerate(
                zip(coordinate_batch, valid_shapes)
            ):
                weight = blend_weight[:valid_height, :valid_width]
                logit_sum[:, y:y + valid_height, x:x + valid_width] += (
                    tile_logits[tile_index, :, :valid_height, :valid_width]
                    * weight.unsqueeze(0)
                )
                weight_sum[y:y + valid_height, x:x + valid_width] += weight

        if not bool(torch.all(weight_sum > 0)):
            raise RuntimeError(f"sliding windows did not cover every pixel of {scene['stem']}")
        stitched_logits = logit_sum / weight_sum.unsqueeze(0)
        if not bool(torch.isfinite(stitched_logits).all()):
            raise RuntimeError(f"non-finite stitched logits for {scene['stem']}")
        target = torch.from_numpy(scene["mask"]).unsqueeze(0)
        accumulator.update(
            stitched_logits.unsqueeze(0), target, scene_ids=[scene["scene_id"]]
        )
        if (
            progress == "log" and rank == 0 and log_interval > 0
            and ((local_scene_index + 1) % log_interval == 0
                 or local_scene_index + 1 == len(indices))
        ):
            print(
                f"[{phase_label}] scenes={local_scene_index + 1}/{len(indices)} "
                f"stem={scene['stem']} windows={len(coordinates)}",
                flush=True,
            )

    accumulator.synchronize_between_processes()
    return accumulator.compute()


def evaluate_test(model, test_source, num_classes, device, args, dice_metrics):
    if args.test_inference_mode == "direct":
        return evaluate(
            model,
            test_source,
            num_classes,
            device,
            amp=args.amp,
            hd95_backend=args.hd95_backend,
            dice_metrics=dice_metrics,
            max_batches=args.max_eval_batches,
        )
    if not isinstance(test_source, ConditionedSlidingWindowTestDataset):
        raise TypeError("sliding_window test mode requires a full-scene test dataset")
    return evaluate_sliding_window(
        model,
        test_source,
        num_classes,
        device,
        window_size=args.test_window_size,
        stride=args.test_window_stride,
        tile_batch_size=args.test_window_batch_size,
        blend=args.test_window_blend,
        amp=args.amp,
        hd95_backend=args.hd95_backend,
        dice_metrics=dice_metrics,
        max_scenes=args.max_eval_batches,
        progress=args.progress,
        log_interval=args.log_interval,
        phase_label="Test sliding",
    )


def _scalar_dice(metrics: dict, name: str) -> float:
    value = metrics["Dice"][name]
    if isinstance(value, dict):
        raise ValueError(f"Dice protocol {name!r} is not scalar and cannot be primary")
    return float(value)


def _flatten_dice_history(metrics: dict, prefix: str) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in metrics["Dice"].items():
        if isinstance(value, dict):
            for class_name, class_value in value.items():
                flattened[f"{prefix}_dice_{class_name}"] = float(class_value)
        else:
            flattened[f"{prefix}_dice_{name}"] = float(value)
    return flattened


def _format_dice(metrics: dict) -> str:
    fields: list[str] = []
    for name, value in metrics["Dice"].items():
        if isinstance(value, dict):
            class_text = ", ".join(
                f"{class_name}={class_value:.4f}"
                for class_name, class_value in value.items()
            )
            fields.append(f"{name}[{class_text}]")
        else:
            fields.append(f"{name}={value:.4f}")
    return " | ".join(fields)


def _flatten_iou_history(metrics: dict, prefix: str) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in metrics["IoU_metrics"].items():
        if isinstance(value, dict):
            for class_name, class_value in value.items():
                flattened[f"{prefix}_iou_{class_name}"] = float(class_value)
        else:
            flattened[f"{prefix}_iou_{name}"] = float(value)
    return flattened


def _format_iou(metrics: dict) -> str:
    fields: list[str] = []
    for name, value in metrics["IoU_metrics"].items():
        if isinstance(value, dict):
            class_text = ", ".join(
                f"{class_name}={class_value:.4f}"
                for class_name, class_value in value.items()
            )
            fields.append(f"{name}[{class_text}]")
        else:
            fields.append(f"{name}={value:.4f}")
    return " | ".join(fields)


def _parse_positive_int_tuple(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _resolve_class_weights(args, train_dataset) -> torch.Tensor | None:
    weighted_loss = args.segmentation_loss.startswith("weighted_")
    if args.class_weight_mode == "none":
        if weighted_loss:
            raise ValueError(
                "weighted segmentation loss requires a non-'none' class-weight mode"
            )
        return None
    if not weighted_loss:
        raise ValueError(
            "class weights are only used by weighted_ce_dice variants"
        )
    if args.class_weight_mode == "manual":
        try:
            values = [float(item.strip()) for item in args.class_weights.split(",")]
        except ValueError as error:
            raise ValueError("--class-weights must be comma-separated floats") from error
        if len(values) != args.num_classes or any(value <= 0 for value in values):
            raise ValueError(
                "manual class weights require one positive value per class"
            )
        weights = torch.tensor(values, dtype=torch.float32)
    else:
        counts = train_dataset.class_pixel_counts(args.num_classes).to(torch.float64)
        if torch.any(counts <= 0):
            raise ValueError(f"cannot derive class weights from counts {counts.tolist()}")
        weights = (
            counts.rsqrt()
            if args.class_weight_mode == "inverse_sqrt"
            else counts.reciprocal()
        ).to(torch.float32)
    return weights / weights.mean()


def _build_optimizer(model: ConditionedFinetuneModel, args):
    staged = args.freeze_backbone_epochs > 0
    differential = (
        args.backbone_lr_multiplier != 1.0 or args.head_lr_multiplier != 1.0
    )
    if not staged and not differential:
        # Historical path: preserve the original parameter iterator exactly.
        return torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = list(model.seg_head.parameters())
    groups = [
        {
            "params": backbone_parameters,
            "lr": args.lr * args.backbone_lr_multiplier,
            "name": "backbone",
        },
        {
            "params": head_parameters,
            "lr": args.lr * args.head_lr_multiplier,
            "name": "segmentation_head",
        },
    ]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def main():
    a = get_args()
    if a.freeze_backbone and a.freeze_backbone_epochs > 0:
        raise ValueError(
            "--freeze-backbone and --freeze-backbone-epochs cannot be used together"
        )
    if a.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs must be nonnegative")
    if a.backbone_lr_multiplier <= 0 or a.head_lr_multiplier <= 0:
        raise ValueError("optimizer LR multipliers must be positive")
    if a.augmentation_copies < 1 or a.augmentation_copies > 8:
        raise ValueError("--augmentation-copies must be in [1,8]")
    if not a.augment and a.augmentation_copies != 1:
        raise ValueError("--augmentation-copies must be 1 when augmentation is disabled")
    if a.segmentation_head == "h3_multiscale_aux" and a.aux_loss_weight <= 0:
        raise ValueError("h3_multiscale_aux requires --aux-loss-weight > 0")
    if a.endmember_scope == "scene" and not a.scene_endmember_root:
        raise ValueError("scene endmember scope requires --scene-endmember-root")
    aspp_rates = _parse_positive_int_tuple(a.aspp_rates, "--aspp-rates")
    if a.scene_val_root:
        if a.scene_val_interval <= 0:
            raise ValueError("--scene-val-interval must be positive when --scene-val-root is set")
        if a.scene_val_window_size <= 0:
            raise ValueError("--scene-val-window-size must be positive")
        if (
            a.scene_val_window_stride <= 0
            or a.scene_val_window_stride > a.scene_val_window_size
        ):
            raise ValueError(
                "--scene-val-window-stride must be positive and no larger than window size"
            )
        if a.scene_val_window_batch_size <= 0:
            raise ValueError("--scene-val-window-batch-size must be positive")
        if a.scene_val_window_size % a.patch_size:
            raise ValueError(
                f"scene validation window size {a.scene_val_window_size} must be "
                f"divisible by model spatial patch size {a.patch_size}"
            )
    if a.test_inference_mode == "sliding_window":
        if a.test_window_size <= 0:
            raise ValueError("--test-window-size must be positive")
        if a.test_window_stride <= 0 or a.test_window_stride > a.test_window_size:
            raise ValueError(
                "--test-window-stride must be positive and no larger than window size"
            )
        if a.test_window_batch_size <= 0:
            raise ValueError("--test-window-batch-size must be positive")
        if a.test_window_size % a.patch_size:
            raise ValueError(
                f"test window size {a.test_window_size} must be divisible by "
                f"model spatial patch size {a.patch_size}"
            )
    dice_metric_names = list(normalize_dice_metric_names(a.dice_metrics))
    primary_candidates = normalize_dice_metric_names([a.primary_dice_metric])
    primary_dice_metric = primary_candidates[0]
    if primary_dice_metric == DICE_CLASSWISE:
        print(
            f"[Finetune] primary Dice {DICE_CLASSWISE!r} is non-scalar; falling back "
            f"to {DICE_BATCH_ALLCLASS_MACRO!r}.",
            flush=True,
        )
        primary_dice_metric = DICE_BATCH_ALLCLASS_MACRO
    if primary_dice_metric not in dice_metric_names:
        dice_metric_names.append(primary_dice_metric)
    # Always report the two dataset-global frequency-weighted protocols.  They
    # are supplementary only: the configured primary metric and checkpoint
    # selection semantics remain unchanged.
    for supplementary_metric in (
        DICE_GLOBAL_FREQUENCY_WEIGHTED_FG,
        DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS,
    ):
        if supplementary_metric not in dice_metric_names:
            dice_metric_names.append(supplementary_metric)
    dice_metric_names_tuple = tuple(dice_metric_names)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    seed = a.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(a.save_dir, exist_ok=True)
    if rank == 0:
        with open(
            os.path.join(a.save_dir, "run_config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(vars(a), handle, ensure_ascii=False, indent=2)
        print(f"[Finetune] HD95 backend: {a.hd95_backend}", flush=True)
        print(
            f"[Finetune] Dice metrics: {', '.join(dice_metric_names_tuple)}; "
            f"primary={primary_dice_metric}",
            flush=True,
        )
        print(
            f"[Finetune] IoU metrics: {', '.join(IOU_METRIC_NAMES)}; "
            "legacy IoU=batch_allclass_macro",
            flush=True,
        )
        print(
            f"[Finetune] DataLoader: workers/rank={a.workers}, "
            f"persistent={a.persistent_workers and a.workers > 0}, "
            f"prefetch_factor={a.prefetch_factor if a.workers > 0 else 'n/a'}, "
            f"distributed_validation={a.distributed_validation}",
            flush=True,
        )
        if a.test_root:
            sliding_details = (
                f", window={a.test_window_size}, stride={a.test_window_stride}, "
                f"tile_batch={a.test_window_batch_size}, blend={a.test_window_blend}"
                if a.test_inference_mode == "sliding_window" else ""
            )
            print(
                f"[Finetune] Test inference: {a.test_inference_mode}{sliding_details}",
                flush=True,
            )
        if a.scene_val_root:
            print(
                "[Finetune] Full-scene validation: "
                f"root={a.scene_val_root}, interval={a.scene_val_interval}, "
                f"window={a.scene_val_window_size}, "
                f"stride={a.scene_val_window_stride}, "
                f"tile_batch={a.scene_val_window_batch_size}, "
                f"blend={a.scene_val_window_blend}; checkpoint selector=scene",
                flush=True,
            )
    data_kwargs = dict(
        patch_size=a.patch_size,
        spectral_patch_size=a.spectral_patch_size,
        nmf_k=a.nmf_k,
        nmf_l1=a.nmf_l1,
        nmf_l2=a.nmf_l2,
        nmf_l3=a.nmf_l3,
        nmf_simplex=a.nmf_simplex,
        nmf_lam_e=a.nmf_lam_e,
        nmf_e_clamp_max=a.nmf_e_clamp_max,
        wavelength_file=a.wavelength_file,
        allow_index_wavelengths=a.allow_index_wavelengths,
        od_max=a.od_max,
        augment=a.augment,
        augmentation_copies=a.augmentation_copies,
        augmentation_seed=a.seed,
        augmentation_policy=a.augmentation_policy,
        augmentation_probability=a.augmentation_probability,
        affine_rotation_degrees=a.affine_rotation_degrees,
        affine_scale_delta=a.affine_scale_delta,
        affine_translate_fraction=a.affine_translate_fraction,
        perspective_scale=a.perspective_scale,
        augmentation_padding_mode=a.augmentation_padding_mode,
        endmember_scope=a.endmember_scope,
        scene_endmember_root=a.scene_endmember_root,
    )
    loader_result = build_conditioned_finetune_loaders(
        a.train_root,
        a.val_root,
        a.test_root,
        a.batch_size,
        a.workers,
        True,
        test_inference_mode=a.test_inference_mode,
        scene_val_root=a.scene_val_root,
        persistent_workers=a.persistent_workers,
        prefetch_factor=a.prefetch_factor,
        distributed_validation=a.distributed_validation,
        validation_max_batches=a.max_eval_batches,
        **data_kwargs,
    )
    if a.scene_val_root:
        train, val, scene_val, test, sampler = loader_result
    else:
        train, val, test, sampler = loader_result
        scene_val = None
    cfg = ConditionedModelConfig(
        patch_size=a.patch_size,
        spectral_patch_size=a.spectral_patch_size,
        embed_dim=a.embed_dim,
        vit_depth=a.vit_depth,
        vit_heads=a.vit_heads,
        mlp_ratio=a.mlp_ratio,
        dropout=a.dropout,
        cnn_stem_ch=a.cnn_stem_ch,
        cnn_spectral_agg=a.cnn_spectral_agg,
        fusion_heads=a.fusion_heads,
        feature_dim=a.feature_dim,
        decoder_mid_ch=a.decoder_mid_ch,
        residual_hidden_dim=a.residual_hidden_dim,
        ridge_lambda=a.ridge_lambda,
        confidence_temperature=a.confidence_temperature,
        alpha_min=a.alpha_min,
        alpha_extra=a.alpha_extra,
        od_max=a.od_max,
    )
    model = ConditionedFinetuneModel(
        a.num_classes,
        cfg,
        a.pretrain_ckpt,
        a.freeze_backbone,
        segmentation_head=a.segmentation_head,
        head_hidden_channels=a.head_hidden_channels,
        head_projection_channels=a.head_projection_channels,
        head_dropout=a.head_dropout,
        aspp_rates=aspp_rates,
    ).cuda()
    # SyncBatchNorm + broadcast_buffers=False：与 LoTS-Net 保持一致，
    # 用 BN 自身的 all_reduce 同步统计量，避免 DDP 在每次 forward（含验证）
    # 都额外做一次 buffer 广播，降低与 MONAI 验证阶段大量小算子交织时的
    # CUDA/NCCL 状态风险。
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[local],
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    if a.eval_only_checkpoint:
        state = (
            torch.load(
                a.eval_only_checkpoint, map_location="cpu", weights_only=False
            )
            if rank == 0
            else None
        )
        shared_state = [state]
        dist.broadcast_object_list(shared_state, src=0)
        model.module.load_state_dict(shared_state[0], strict=True)
        if test is None:
            raise ValueError("--eval-only-checkpoint requires --test-root")
        test_metrics = evaluate_test(
            model, test, a.num_classes, local, a, dice_metric_names_tuple
        )
        if rank == 0:
            print(
                f"[Eval-only] Test primary Dice ({primary_dice_metric}): "
                f"{_scalar_dice(test_metrics, primary_dice_metric):.4f}  "
                f"IoU: {test_metrics['IoU']:.4f}  HD95: {test_metrics['HD95']:.4f}",
                flush=True,
            )
            print(f"[Eval-only] Test Dice: {_format_dice(test_metrics)}", flush=True)
            print(f"[Eval-only] Test IoU: {_format_iou(test_metrics)}", flush=True)
            with open(
                os.path.join(a.save_dir, "eval_only_metrics.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "checkpoint": a.eval_only_checkpoint,
                        "primary_dice_metric": primary_dice_metric,
                        "dice_metrics": list(dice_metric_names_tuple),
                        "iou_metrics": list(IOU_METRIC_NAMES),
                        "test_inference": {
                            "mode": a.test_inference_mode,
                            "window_size": a.test_window_size,
                            "window_stride": a.test_window_stride,
                            "window_batch_size": a.test_window_batch_size,
                            "blend": a.test_window_blend,
                        },
                        "metrics": test_metrics,
                    },
                    handle,
                    indent=2,
                    allow_nan=True,
                )
        dist.destroy_process_group()
        return
    inner_model = model.module
    optimizer = _build_optimizer(inner_model, a)
    scheduler = build_cosine_scheduler(
        optimizer, a.epochs, a.warmup_epochs, len(train), a.lr, a.min_lr
    )
    class_weights = _resolve_class_weights(a, train.dataset)
    criterion = build_segmentation_criterion(
        a.num_classes,
        loss_type=a.segmentation_loss,
        class_weights=class_weights,
        ce_weight=a.ce_loss_weight,
        dice_weight=a.dice_loss_weight,
        focal_gamma=a.focal_gamma,
        boundary_weight=a.boundary_loss_weight,
        auxiliary_weight=a.aux_loss_weight,
    ).cuda()
    if rank == 0:
        print(
            f"[Finetune] head={a.segmentation_head} loss={a.segmentation_loss} "
            f"augment={a.augment}/{a.augmentation_policy}/copies{a.augmentation_copies}",
            flush=True,
        )
        print(
            f"[Finetune] optimizer backbone_lr_multiplier={a.backbone_lr_multiplier:g} "
            f"head_lr_multiplier={a.head_lr_multiplier:g} "
            f"freeze_backbone_epochs={a.freeze_backbone_epochs}",
            flush=True,
        )
        if class_weights is not None:
            print(f"[Finetune] class weights: {class_weights.tolist()}", flush=True)
    # bfloat16 autocast 且不使用 GradScaler：与 LoTS-Net 训练脚本一致。
    # fp16 + GradScaler 每个 step 都需要同步读取 found_inf_per_device，
    # 这个高频强制同步点容易把 MONAI HD95 在验证阶段可能引入的异步 CUDA
    # 异常，在下一个 step 精确地暴露出来。bf16 动态范围更大，无需损失缩放，
    # 从而去掉了这个高频陷阱。
    best_dice = 0.0
    best_patch_dice = 0.0
    no_improve = 0
    best_path = os.path.join(a.save_dir, "ckpt_best.pth")
    best_patch_path = os.path.join(a.save_dir, "ckpt_best_patch.pth")
    best_scene_path = os.path.join(a.save_dir, "ckpt_best_scene.pth")
    latest_path = os.path.join(a.save_dir, "ckpt_last.pth")
    curve_monitor = (
        FinetuneCurveMonitor(
            a.save_dir,
            build_segmentation_curve_groups(
                dice_metric_names_tuple,
                a.num_classes,
                iou_metric_names=IOU_METRIC_NAMES,
            ),
        )
        if rank == 0
        else None
    )
    # 窗口最佳：window_key -> 该窗口内已见过的最高 Val Dice（仅 rank0 使用）
    window_best_dice: dict[str, float] = {}
    for epoch in range(1, a.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        if hasattr(train.dataset, "set_epoch"):
            train.dataset.set_epoch(epoch)
        staged_backbone_frozen = epoch <= a.freeze_backbone_epochs
        if staged_backbone_frozen:
            # Keep DDP's parameter set fixed: gradients are synchronized, then
            # cleared before optimizer.step.  Evaluation mode also prevents BN
            # running-stat updates during the frozen stage.
            inner_model.backbone.eval()
        total = torch.zeros(2, device=local)
        t0 = time.time()
        iterable = enumerate(train)
        if a.progress == "tqdm" and rank == 0:
            iterable = tqdm(
                iterable, total=len(train), desc=f"Train {epoch:04d}", leave=False
            )
        for step, batch in iterable:
            if a.max_train_batches > 0 and step >= a.max_train_batches:
                break
            batch.pop("stem", None)
            batch.pop("scene_id", None)
            batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            seg = batch.pop("seg")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                prediction = model(batch)
                loss, _ = criterion(prediction, seg)
            loss.backward()
            if staged_backbone_frozen:
                for parameter in inner_model.backbone.parameters():
                    parameter.grad = None
            if a.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            optimizer.step()
            scheduler.step()
            # Keep epoch-loss accumulation on device.  Converting the loss to a
            # Python float here forced a GPU-to-CPU synchronization every step;
            # only interval logging and the epoch summary actually need a host
            # value.  This does not participate in backward or optimizer state.
            total[0].add_(loss.detach())
            total[1].add_(1)
            if (
                a.progress == "log"
                and rank == 0
                and a.log_interval > 0
                and (step + 1) % a.log_interval == 0
            ):
                print(
                    f'[{time.strftime("%F %T")}] epoch={epoch} '
                    f"step={step+1}/{len(train)} loss={float(loss.detach()):.5f}",
                    flush=True,
                )
        dist.all_reduce(total)
        val_metrics = evaluate(
            model,
            val,
            a.num_classes,
            local,
            amp=a.amp,
            hd95_backend=a.hd95_backend,
            dice_metrics=dice_metric_names_tuple,
            max_batches=a.max_eval_batches,
        )
        dice = _scalar_dice(val_metrics, primary_dice_metric)
        iou = float(val_metrics["IoU"])
        hd = float(val_metrics["HD95"])
        run_scene_val = bool(
            scene_val is not None
            and (epoch % a.scene_val_interval == 0 or epoch == a.epochs)
        )
        scene_val_metrics = None
        if run_scene_val:
            scene_val_metrics = evaluate_sliding_window(
                model,
                scene_val,
                a.num_classes,
                local,
                window_size=a.scene_val_window_size,
                stride=a.scene_val_window_stride,
                tile_batch_size=a.scene_val_window_batch_size,
                blend=a.scene_val_window_blend,
                amp=a.amp,
                hd95_backend=a.hd95_backend,
                dice_metrics=dice_metric_names_tuple,
                max_scenes=a.max_eval_batches,
                progress=a.progress,
                log_interval=a.log_interval,
                phase_label="Scene val sliding",
            )
        stop = torch.zeros(1, dtype=torch.int32, device=local)
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = float((total[0] / total[1].clamp(min=1)).item())
            assert curve_monitor is not None
            history_values = {
                "train_loss": train_loss,
                "val_dice": dice,
                "val_dice_primary": dice,
                "val_iou": iou,
                "val_hd95": hd,
                "learning_rate": current_lr,
            }
            history_values.update(_flatten_dice_history(val_metrics, "val"))
            history_values.update(_flatten_iou_history(val_metrics, "val"))
            if scene_val_metrics is not None:
                scene_dice = _scalar_dice(scene_val_metrics, primary_dice_metric)
                history_values.update(
                    {
                        "val_scene_dice_primary": scene_dice,
                        "val_scene_iou": float(scene_val_metrics["IoU"]),
                        "val_scene_hd95": float(scene_val_metrics["HD95"]),
                    }
                )
                history_values.update(
                    _flatten_dice_history(scene_val_metrics, "val_scene")
                )
                history_values.update(
                    _flatten_iou_history(scene_val_metrics, "val_scene")
                )
            curve_monitor.record(epoch, history_values)
            print(
                f'[{time.strftime("%F %T")}] Epoch {epoch}/{a.epochs} '
                f"loss={train_loss:.5f} | "
                f"Val primary Dice ({primary_dice_metric}): {dice:.4f} | "
                f"Val IoU: {iou:.4f} | Val HD95: {hd:.4f} | "
                f"LR: {current_lr:.1e} | time={time.time()-t0:.1f}s",
                flush=True,
            )
            print(f"[Val Dice] {_format_dice(val_metrics)}", flush=True)
            print(f"[Val IoU] {_format_iou(val_metrics)}", flush=True)
            if scene_val_metrics is not None:
                print(
                    f"[Scene Val] primary Dice ({primary_dice_metric}): "
                    f"{scene_dice:.4f} | IoU: {scene_val_metrics['IoU']:.4f} | "
                    f"HD95: {scene_val_metrics['HD95']:.4f}",
                    flush=True,
                )
                print(
                    f"[Scene Val Dice] {_format_dice(scene_val_metrics)}", flush=True
                )
                print(
                    f"[Scene Val IoU] {_format_iou(scene_val_metrics)}", flush=True
                )
            state = (model.module if isinstance(model, DDP) else model).state_dict()
            torch.save(state, latest_path)
            if scene_val is not None:
                if dice > best_patch_dice:
                    best_patch_dice = dice
                    torch.save(state, best_patch_path)
                    print(
                        f">>> New Best Patch-Val Model Saved! {primary_dice_metric}: "
                        f"{best_patch_dice:.4f}",
                        flush=True,
                    )
                selector_dice = scene_dice if scene_val_metrics is not None else None
            else:
                selector_dice = dice
            selector_improved = False
            if selector_dice is not None:
                if selector_dice > best_dice:
                    selector_improved = True
                    best_dice = selector_dice
                    no_improve = 0
                    torch.save(state, best_path)
                    if scene_val is not None:
                        torch.save(state, best_scene_path)
                    print(
                        f">>> New Best {'Scene-Val ' if scene_val is not None else ''}"
                        f"Model Saved! {primary_dice_metric}: {best_dice:.4f}",
                        flush=True,
                    )
                else:
                    no_improve += 1
            if a.early_stop and selector_dice is not None:
                if not selector_improved:
                    print(
                        f"[EarlyStop] No improvement for {no_improve}/{a.patience} "
                        f"validation checks. Best {primary_dice_metric}: {best_dice:.4f}",
                        flush=True,
                    )
            if epoch % a.save_interval == 0:
                torch.save(
                    state, os.path.join(a.save_dir, f"ckpt_epoch{epoch:04d}.pth")
                )
            if a.best_val_interval > 0:
                window_idx = (epoch - 1) // a.best_val_interval
                window_start = window_idx * a.best_val_interval + 1
                window_end = window_start + a.best_val_interval - 1
                window_key = f"{window_start:04d}-{window_end:04d}"
                window_selector_dice = (
                    scene_dice if scene_val_metrics is not None
                    else dice if scene_val is None
                    else None
                )
                if (
                    window_selector_dice is not None
                    and window_selector_dice > window_best_dice.get(window_key, -1.0)
                ):
                    window_best_dice[window_key] = window_selector_dice
                    window_path = os.path.join(
                        a.save_dir, f"ckpt_window{window_key}_best.pth"
                    )
                    torch.save(state, window_path)
                    print(
                        f">>> [Window {window_key}] New best {primary_dice_metric}: "
                        f"{window_selector_dice:.4f} "
                        f"at epoch {epoch}",
                        flush=True,
                    )
            if a.early_stop and no_improve >= a.patience:
                stop.fill_(1)
        dist.broadcast(stop, 0)
        if stop.item():
            break
    if test:
        dist.barrier()
        inner = model.module

        def _load_and_test(ckpt_path: str) -> dict:
            state = (
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if rank == 0
                else None
            )
            obj = [state]
            dist.broadcast_object_list(obj, src=0)
            inner.load_state_dict(obj[0])
            return evaluate_test(
                model, test, a.num_classes, local, a, dice_metric_names_tuple
            )

        test_results: dict[str, dict] = {}

        # 先逐个测试每个窗口内验证集最佳的模型，避免全局最优过早出现时，
        # 后续 epoch 学到的信息被完全忽视而无法在测试集上体现。
        if a.best_val_interval > 0:
            if rank == 0:
                window_paths = sorted(
                    glob.glob(os.path.join(a.save_dir, "ckpt_window*_best.pth"))
                )
            else:
                window_paths = None
            obj = [window_paths]
            dist.broadcast_object_list(obj, src=0)
            window_paths = obj[0]
            for window_path in window_paths:
                label = os.path.splitext(os.path.basename(window_path))[0]
                test_metrics = _load_and_test(window_path)
                test_results[label] = test_metrics
                if rank == 0:
                    print(
                        f"[{label}] Test primary Dice ({primary_dice_metric}): "
                        f"{_scalar_dice(test_metrics, primary_dice_metric):.4f}  "
                        f"IoU: {test_metrics['IoU']:.4f}  "
                        f"HD95: {test_metrics['HD95']:.4f}",
                        flush=True,
                    )
                    print(
                        f"[{label}] Test Dice: {_format_dice(test_metrics)}",
                        flush=True,
                    )
                    print(
                        f"[{label}] Test IoU: {_format_iou(test_metrics)}",
                        flush=True,
                    )

        best_test_metrics = _load_and_test(best_path)
        test_results["ckpt_best"] = best_test_metrics
        if rank == 0:
            print(
                f"Test primary Dice ({primary_dice_metric}): "
                f"{_scalar_dice(best_test_metrics, primary_dice_metric):.4f}  "
                f"IoU: {best_test_metrics['IoU']:.4f}  "
                f"HD95: {best_test_metrics['HD95']:.4f}",
                flush=True,
            )
            print(f"Test Dice: {_format_dice(best_test_metrics)}", flush=True)
            print(f"Test IoU: {_format_iou(best_test_metrics)}", flush=True)
            with open(
                os.path.join(a.save_dir, "test_metrics.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "primary_dice_metric": primary_dice_metric,
                        "dice_metrics": list(dice_metric_names_tuple),
                        "iou_metrics": list(IOU_METRIC_NAMES),
                        "checkpoint_selection": (
                            "complete_scene_validation"
                            if scene_val is not None
                            else "patch_validation"
                        ),
                        "scene_validation": (
                            {
                                "root": a.scene_val_root,
                                "interval": a.scene_val_interval,
                                "window_size": a.scene_val_window_size,
                                "window_stride": a.scene_val_window_stride,
                                "window_batch_size": a.scene_val_window_batch_size,
                                "blend": a.scene_val_window_blend,
                            }
                            if scene_val is not None
                            else None
                        ),
                        "test_inference": {
                            "mode": a.test_inference_mode,
                            "window_size": a.test_window_size,
                            "window_stride": a.test_window_stride,
                            "window_batch_size": a.test_window_batch_size,
                            "blend": a.test_window_blend,
                        },
                        "results": test_results,
                    },
                    handle,
                    indent=2,
                    allow_nan=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
