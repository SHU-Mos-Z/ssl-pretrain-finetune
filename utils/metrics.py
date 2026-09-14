"""评估指标（分割 + 重建监控）。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import warnings
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from monai.metrics import compute_hausdorff_distance

    _MONAI_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境未安装 monai 时优雅降级
    _MONAI_AVAILABLE = False

HD95_BACKEND_CHOICES = ("scipy", "monai")

DICE_BATCH_ALLCLASS_MACRO = "batch_allclass_macro"
DICE_BATCH_FG_MACRO = "batch_fg_macro"
DICE_FG_BINARY_SCENE = "fg_binary_scene"
DICE_MICRO_FG_SCENE = "micro_fg_scene"
DICE_WEIGHTED_FG_SCENE = "weighted_fg_scene"
DICE_MACRO_FG_SCENE = "macro_fg_scene"
DICE_CLASSWISE = "classwise"
DICE_GLOBAL_FG = "global_fg"
DICE_GLOBAL_FREQUENCY_WEIGHTED_FG = "global_frequency_weighted_fg"
DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS = "global_frequency_weighted_all_class"
DICE_METRIC_NAMES = (
    DICE_BATCH_ALLCLASS_MACRO,
    DICE_BATCH_FG_MACRO,
    DICE_FG_BINARY_SCENE,
    DICE_MICRO_FG_SCENE,
    DICE_WEIGHTED_FG_SCENE,
    DICE_MACRO_FG_SCENE,
    DICE_CLASSWISE,
    DICE_GLOBAL_FG,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_FG,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS,
)

# IoU uses the same aggregation protocol names as Dice.  Keeping the names
# aligned makes the two families directly comparable in logs and JSON output.
IOU_BATCH_ALLCLASS_MACRO = DICE_BATCH_ALLCLASS_MACRO
IOU_BATCH_FG_MACRO = DICE_BATCH_FG_MACRO
IOU_FG_BINARY_SCENE = DICE_FG_BINARY_SCENE
IOU_MICRO_FG_SCENE = DICE_MICRO_FG_SCENE
IOU_WEIGHTED_FG_SCENE = DICE_WEIGHTED_FG_SCENE
IOU_MACRO_FG_SCENE = DICE_MACRO_FG_SCENE
IOU_CLASSWISE = DICE_CLASSWISE
IOU_GLOBAL_FG = DICE_GLOBAL_FG
IOU_GLOBAL_FREQUENCY_WEIGHTED_FG = DICE_GLOBAL_FREQUENCY_WEIGHTED_FG
IOU_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS = (
    DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS
)
IOU_METRIC_NAMES = DICE_METRIC_NAMES

_DICE_NAME_ALIASES = {
    "dice": DICE_BATCH_ALLCLASS_MACRO,
    "default": DICE_BATCH_ALLCLASS_MACRO,
    "current": DICE_BATCH_ALLCLASS_MACRO,
    "dice_batch_fg_macro": DICE_BATCH_FG_MACRO,
    "dice_fg_binary_scene": DICE_FG_BINARY_SCENE,
    "dice_micro_fg_scene": DICE_MICRO_FG_SCENE,
    "dice_weighted_fg_scene": DICE_WEIGHTED_FG_SCENE,
    "dice_macro_fg_scene": DICE_MACRO_FG_SCENE,
    "dice_classwise": DICE_CLASSWISE,
    "dice_global_fg": DICE_GLOBAL_FG,
    "dice_global_frequency_weighted_fg": DICE_GLOBAL_FREQUENCY_WEIGHTED_FG,
    "dice_global_frequency_weighted_all_class": (
        DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS
    ),
}


def normalize_dice_metric_names(
    names: Sequence[str] | str | None,
    *,
    warn_invalid: bool = True,
) -> tuple[str, ...]:
    """Normalize requested Dice protocols and preserve their requested order.

    Empty input uses the historical metric. Invalid names are ignored; when no
    valid name remains, the historical metric is used as a safe fallback.
    """
    if isinstance(names, str):
        raw_names = [item.strip() for item in names.split(",")]
    elif names is None:
        raw_names = []
    else:
        raw_names = [str(item).strip() for item in names]
    raw_names = [item for item in raw_names if item]
    if not raw_names:
        return (DICE_BATCH_ALLCLASS_MACRO,)

    valid: list[str] = []
    invalid: list[str] = []
    for raw_name in raw_names:
        key = raw_name.lower().replace("-", "_")
        key = _DICE_NAME_ALIASES.get(key, key)
        if key not in DICE_METRIC_NAMES:
            invalid.append(raw_name)
            continue
        if key not in valid:
            valid.append(key)
    if invalid and warn_invalid:
        warnings.warn(
            "Ignoring unknown Dice metric name(s): " + ", ".join(invalid),
            RuntimeWarning,
            stacklevel=2,
        )
    if not valid:
        if warn_invalid:
            warnings.warn(
                f"No valid Dice metric was requested; falling back to "
                f"{DICE_BATCH_ALLCLASS_MACRO!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return (DICE_BATCH_ALLCLASS_MACRO,)
    return tuple(valid)


def infer_segmentation_scene_id(stem: str) -> str:
    """Infer the source-scene identifier from common preprocessing suffixes.

    The current datasets do not retain patch coordinates in each sample. This
    function therefore groups patch pixels by their source stem; it does *not*
    reconstruct overlapping patches into an original image canvas.
    """
    value = str(stem)
    for pattern in (r"_p\d+_\d+$", r"_roi_\d+$", r"_p\d+$"):
        reduced = re.sub(pattern, "", value)
        if reduced != value:
            return reduced
    return value


@dataclass
class _SceneDiceCounts:
    intersection: np.ndarray
    predicted: np.ndarray
    target: np.ndarray
    binary_intersection: float = 0.0
    binary_predicted: float = 0.0
    binary_target: float = 0.0

    @classmethod
    def zeros(cls, num_classes: int) -> "_SceneDiceCounts":
        return cls(
            intersection=np.zeros(num_classes, dtype=np.float64),
            predicted=np.zeros(num_classes, dtype=np.float64),
            target=np.zeros(num_classes, dtype=np.float64),
        )

    def merge_(self, other: "_SceneDiceCounts") -> None:
        self.intersection += other.intersection
        self.predicted += other.predicted
        self.target += other.target
        self.binary_intersection += other.binary_intersection
        self.binary_predicted += other.binary_predicted
        self.binary_target += other.binary_target


def _dice_or_nan(intersection: float, predicted: float, target: float) -> float:
    denominator = float(predicted + target)
    if denominator <= 0.0:
        return float("nan")
    return float(2.0 * intersection / denominator)


def _iou_or_nan(intersection: float, predicted: float, target: float) -> float:
    denominator = float(predicted + target - intersection)
    if denominator <= 0.0:
        return float("nan")
    return float(intersection / denominator)


def _global_frequency_weighted_overlap(
    state: "SegmentationMetricAccumulator",
    overlap_function: Callable[[float, float, float], float],
    *,
    include_background: bool,
) -> float:
    """GT-frequency-weighted overlap from dataset-global class counts."""
    start = 0 if include_background else 1
    target_counts = state.global_target[start:]
    total_target = float(target_counts.sum())
    if total_target <= 0.0:
        return float("nan")
    value = 0.0
    for class_index in range(start, state.num_classes):
        weight = float(state.global_target[class_index] / total_target)
        if weight <= 0.0:
            continue
        overlap = overlap_function(
            float(state.global_intersection[class_index]),
            float(state.global_predicted[class_index]),
            float(state.global_target[class_index]),
        )
        value += weight * (0.0 if not np.isfinite(overlap) else overlap)
    return float(value)


def dice_batch_allclass_macro(state: "SegmentationMetricAccumulator") -> float:
    """Historical metric: sample-count-weighted mean of batch all-class macro Dice."""
    if state.batch_sample_count <= 0:
        return float("nan")
    return float(state.batch_dice_sum / state.batch_sample_count)


def dice_batch_fg_macro(state: "SegmentationMetricAccumulator") -> float:
    """Sample-count-weighted batch macro Dice over union-present foreground classes.

    Background is excluded. A foreground class absent from both prediction and
    target in a batch is skipped; one-sided presence remains an evaluable error
    and receives Dice 0. A batch with no evaluable foreground class is skipped.
    """
    if state.batch_fg_sample_count <= 0:
        return float("nan")
    return float(state.batch_fg_dice_sum / state.batch_fg_sample_count)


def dice_fg_binary_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise binary Dice after collapsing all foreground labels."""
    values = [
        _dice_or_nan(
            scene.binary_intersection,
            scene.binary_predicted,
            scene.binary_target,
        )
        for scene in state.scene_counts.values()
    ]
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def dice_micro_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise foreground micro Dice, preserving foreground classes."""
    values: list[float] = []
    for scene in state.scene_counts.values():
        values.append(
            _dice_or_nan(
                float(scene.intersection[1:].sum()),
                float(scene.predicted[1:].sum()),
                float(scene.target[1:].sum()),
            )
        )
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def dice_weighted_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise foreground Dice weighted by each class's GT area."""
    scene_values: list[float] = []
    for scene in state.scene_counts.values():
        total_target = float(scene.target[1:].sum())
        if total_target <= 0.0:
            continue
        value = 0.0
        for class_index in range(1, state.num_classes):
            weight = float(scene.target[class_index] / total_target)
            if weight <= 0.0:
                continue
            class_dice = _dice_or_nan(
                float(scene.intersection[class_index]),
                float(scene.predicted[class_index]),
                float(scene.target[class_index]),
            )
            value += weight * (0.0 if not np.isfinite(class_dice) else class_dice)
        scene_values.append(value)
    return float(np.mean(scene_values)) if scene_values else float("nan")


def dice_macro_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise macro Dice over non-empty foreground classes.

    A class absent from both prediction and target is skipped. A class present
    on only one side receives Dice 0. A scene with no evaluable foreground
    class is skipped.
    """
    scene_values: list[float] = []
    for scene in state.scene_counts.values():
        class_values = [
            _dice_or_nan(
                float(scene.intersection[class_index]),
                float(scene.predicted[class_index]),
                float(scene.target[class_index]),
            )
            for class_index in range(1, state.num_classes)
        ]
        valid = [value for value in class_values if np.isfinite(value)]
        if valid:
            scene_values.append(float(np.mean(valid)))
    return float(np.mean(scene_values)) if scene_values else float("nan")


def dice_classwise(state: "SegmentationMetricAccumulator") -> dict[str, float]:
    """Global Dice for every class, including class 0 (background)."""
    return {
        f"class_{class_index}": _dice_or_nan(
            float(state.global_intersection[class_index]),
            float(state.global_predicted[class_index]),
            float(state.global_target[class_index]),
        )
        for class_index in range(state.num_classes)
    }


def dice_global_fg(state: "SegmentationMetricAccumulator") -> float:
    """Macro foreground Dice from dataset-global per-class pixel counts."""
    class_values = list(dice_classwise(state).values())[1:]
    valid = [value for value in class_values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def dice_global_frequency_weighted_fg(
    state: "SegmentationMetricAccumulator",
) -> float:
    """Dataset-global foreground Dice weighted by foreground GT pixel frequency."""
    return _global_frequency_weighted_overlap(
        state, _dice_or_nan, include_background=False
    )


def dice_global_frequency_weighted_all_class(
    state: "SegmentationMetricAccumulator",
) -> float:
    """Dataset-global all-class Dice weighted by all-class GT pixel frequency."""
    return _global_frequency_weighted_overlap(
        state, _dice_or_nan, include_background=True
    )


DiceResult = float | dict[str, float]
DiceFunction = Callable[["SegmentationMetricAccumulator"], DiceResult]
DICE_METRIC_FUNCTIONS: dict[str, DiceFunction] = {
    DICE_BATCH_ALLCLASS_MACRO: dice_batch_allclass_macro,
    DICE_BATCH_FG_MACRO: dice_batch_fg_macro,
    DICE_FG_BINARY_SCENE: dice_fg_binary_scene,
    DICE_MICRO_FG_SCENE: dice_micro_fg_scene,
    DICE_WEIGHTED_FG_SCENE: dice_weighted_fg_scene,
    DICE_MACRO_FG_SCENE: dice_macro_fg_scene,
    DICE_CLASSWISE: dice_classwise,
    DICE_GLOBAL_FG: dice_global_fg,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_FG: dice_global_frequency_weighted_fg,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS: (
        dice_global_frequency_weighted_all_class
    ),
}


def iou_batch_allclass_macro(state: "SegmentationMetricAccumulator") -> float:
    """Historical sample-count-weighted mean of batch all-class macro IoU."""
    if state.batch_sample_count <= 0:
        return float("nan")
    return float(state.batch_iou_sum / state.batch_sample_count)


def iou_batch_fg_macro(state: "SegmentationMetricAccumulator") -> float:
    """Sample-count-weighted batch macro IoU over union-present foreground classes."""
    if state.batch_fg_sample_count <= 0:
        return float("nan")
    return float(state.batch_fg_iou_sum / state.batch_fg_sample_count)


def iou_fg_binary_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise binary IoU after collapsing all foreground labels."""
    values = [
        _iou_or_nan(
            scene.binary_intersection,
            scene.binary_predicted,
            scene.binary_target,
        )
        for scene in state.scene_counts.values()
    ]
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def iou_micro_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise foreground micro IoU, preserving foreground classes."""
    values = [
        _iou_or_nan(
            float(scene.intersection[1:].sum()),
            float(scene.predicted[1:].sum()),
            float(scene.target[1:].sum()),
        )
        for scene in state.scene_counts.values()
    ]
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def iou_weighted_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise foreground IoU weighted by each class's GT area."""
    scene_values: list[float] = []
    for scene in state.scene_counts.values():
        total_target = float(scene.target[1:].sum())
        if total_target <= 0.0:
            continue
        value = 0.0
        for class_index in range(1, state.num_classes):
            weight = float(scene.target[class_index] / total_target)
            if weight <= 0.0:
                continue
            class_iou = _iou_or_nan(
                float(scene.intersection[class_index]),
                float(scene.predicted[class_index]),
                float(scene.target[class_index]),
            )
            value += weight * (0.0 if not np.isfinite(class_iou) else class_iou)
        scene_values.append(value)
    return float(np.mean(scene_values)) if scene_values else float("nan")


def iou_macro_fg_scene(state: "SegmentationMetricAccumulator") -> float:
    """Mean scene-wise macro IoU over non-empty foreground classes."""
    scene_values: list[float] = []
    for scene in state.scene_counts.values():
        class_values = [
            _iou_or_nan(
                float(scene.intersection[class_index]),
                float(scene.predicted[class_index]),
                float(scene.target[class_index]),
            )
            for class_index in range(1, state.num_classes)
        ]
        valid = [value for value in class_values if np.isfinite(value)]
        if valid:
            scene_values.append(float(np.mean(valid)))
    return float(np.mean(scene_values)) if scene_values else float("nan")


def iou_classwise(state: "SegmentationMetricAccumulator") -> dict[str, float]:
    """Dataset-global IoU for every class, including class 0 (background)."""
    return {
        f"class_{class_index}": _iou_or_nan(
            float(state.global_intersection[class_index]),
            float(state.global_predicted[class_index]),
            float(state.global_target[class_index]),
        )
        for class_index in range(state.num_classes)
    }


def iou_global_fg(state: "SegmentationMetricAccumulator") -> float:
    """Macro foreground IoU from dataset-global per-class pixel counts."""
    class_values = list(iou_classwise(state).values())[1:]
    valid = [value for value in class_values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def iou_global_frequency_weighted_fg(
    state: "SegmentationMetricAccumulator",
) -> float:
    """Dataset-global foreground IoU weighted by foreground GT pixel frequency."""
    return _global_frequency_weighted_overlap(
        state, _iou_or_nan, include_background=False
    )


def iou_global_frequency_weighted_all_class(
    state: "SegmentationMetricAccumulator",
) -> float:
    """Dataset-global all-class IoU weighted by all-class GT pixel frequency."""
    return _global_frequency_weighted_overlap(
        state, _iou_or_nan, include_background=True
    )


IoUResult = float | dict[str, float]
IoUFunction = Callable[["SegmentationMetricAccumulator"], IoUResult]
IOU_METRIC_FUNCTIONS: dict[str, IoUFunction] = {
    IOU_BATCH_ALLCLASS_MACRO: iou_batch_allclass_macro,
    IOU_BATCH_FG_MACRO: iou_batch_fg_macro,
    IOU_FG_BINARY_SCENE: iou_fg_binary_scene,
    IOU_MICRO_FG_SCENE: iou_micro_fg_scene,
    IOU_WEIGHTED_FG_SCENE: iou_weighted_fg_scene,
    IOU_MACRO_FG_SCENE: iou_macro_fg_scene,
    IOU_CLASSWISE: iou_classwise,
    IOU_GLOBAL_FG: iou_global_fg,
    IOU_GLOBAL_FREQUENCY_WEIGHTED_FG: iou_global_frequency_weighted_fg,
    IOU_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS: (
        iou_global_frequency_weighted_all_class
    ),
}


@dataclass
class SegmentationMetricAccumulator:
    """Accumulate sufficient statistics before evaluating non-linear metrics."""

    num_classes: int
    dice_metrics: Sequence[str] | str | None = None
    class_weights: list[float] | None = None
    hd95_backend: str = "scipy"
    ignore_index: int = -1
    metric_names: tuple[str, ...] = field(init=False)
    global_intersection: np.ndarray = field(init=False)
    global_predicted: np.ndarray = field(init=False)
    global_target: np.ndarray = field(init=False)
    scene_counts: dict[str, _SceneDiceCounts] = field(default_factory=dict)
    batch_dice_sum: float = 0.0
    batch_iou_sum: float = 0.0
    batch_fg_dice_sum: float = 0.0
    batch_fg_iou_sum: float = 0.0
    batch_fg_sample_count: int = 0
    batch_hd95_sum: float = 0.0
    batch_sample_count: int = 0
    _anonymous_scene_offset: int = 0

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.hd95_backend not in HD95_BACKEND_CHOICES:
            raise ValueError(
                f"hd95_backend must be one of {HD95_BACKEND_CHOICES}, "
                f"got {self.hd95_backend!r}"
            )
        self.metric_names = normalize_dice_metric_names(self.dice_metrics)
        self.global_intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.global_predicted = np.zeros(self.num_classes, dtype=np.float64)
        self.global_target = np.zeros(self.num_classes, dtype=np.float64)

    @staticmethod
    def _hard_predictions(pred_or_logits: torch.Tensor) -> torch.Tensor:
        if pred_or_logits.dim() == 4:
            return pred_or_logits.argmax(dim=1)
        if pred_or_logits.dim() == 3:
            return pred_or_logits
        raise ValueError(
            "predictions must have shape [B,C,H,W] or [B,H,W], got "
            f"{tuple(pred_or_logits.shape)}"
        )

    def _counts_for_mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        intersection = np.zeros(self.num_classes, dtype=np.float64)
        predicted = np.zeros(self.num_classes, dtype=np.float64)
        target_area = np.zeros(self.num_classes, dtype=np.float64)
        for class_index in range(self.num_classes):
            pred_c = (pred == class_index) & valid
            target_c = (target == class_index) & valid
            intersection[class_index] = float((pred_c & target_c).sum().item())
            predicted[class_index] = float(pred_c.sum().item())
            target_area[class_index] = float(target_c.sum().item())
        return intersection, predicted, target_area

    def update(
        self,
        pred_or_logits: torch.Tensor,
        target: torch.Tensor,
        *,
        scene_ids: Sequence[str] | None = None,
    ) -> None:
        pred = self._hard_predictions(pred_or_logits).detach()
        target = target.detach()
        if pred.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {tuple(pred.shape)} vs "
                f"{tuple(target.shape)}"
            )
        batch_size = int(target.shape[0])
        if scene_ids is None:
            scene_ids = [
                f"__sample_{self._anonymous_scene_offset + index}"
                for index in range(batch_size)
            ]
        if len(scene_ids) != batch_size:
            raise ValueError(
                f"scene_ids length {len(scene_ids)} does not match batch size {batch_size}"
            )
        self._anonymous_scene_offset += batch_size

        valid = target != self.ignore_index
        intersection, predicted, target_area = self._counts_for_mask(
            pred, target, valid
        )
        self.global_intersection += intersection
        self.global_predicted += predicted
        self.global_target += target_area

        batch_dice = 2.0 * intersection / (predicted + target_area + 1e-5)
        union = predicted + target_area - intersection
        batch_iou = intersection / (union + 1e-5)
        if self.class_weights is not None and len(self.class_weights) == self.num_classes:
            weights = np.asarray(
                list(reversed(self.class_weights)), dtype=np.float64
            )
            weights /= weights.sum()
            mean_batch_dice = float(np.sum(batch_dice * weights))
            mean_batch_iou = float(np.sum(batch_iou * weights))
        else:
            mean_batch_dice = float(np.mean(batch_dice))
            mean_batch_iou = float(np.mean(batch_iou))
        self.batch_dice_sum += mean_batch_dice * batch_size
        self.batch_iou_sum += mean_batch_iou * batch_size
        foreground_evaluable = (predicted[1:] + target_area[1:]) > 0.0
        if np.any(foreground_evaluable):
            foreground_denominator = predicted[1:] + target_area[1:]
            foreground_union = (
                predicted[1:] + target_area[1:] - intersection[1:]
            )
            foreground_dice = (
                2.0 * intersection[1:][foreground_evaluable]
                / foreground_denominator[foreground_evaluable]
            )
            foreground_iou = (
                intersection[1:][foreground_evaluable]
                / foreground_union[foreground_evaluable]
            )
            mean_batch_fg_dice = float(np.mean(foreground_dice))
            mean_batch_fg_iou = float(np.mean(foreground_iou))
            self.batch_fg_dice_sum += mean_batch_fg_dice * batch_size
            self.batch_fg_iou_sum += mean_batch_fg_iou * batch_size
            self.batch_fg_sample_count += batch_size
        self.batch_hd95_sum += (
            compute_hd95(
                pred,
                target,
                self.num_classes,
                backend=self.hd95_backend,
                ignore_index=self.ignore_index,
            )
            * batch_size
        )
        self.batch_sample_count += batch_size

        for sample_index, raw_scene_id in enumerate(scene_ids):
            scene_id = str(raw_scene_id)
            scene = self.scene_counts.setdefault(
                scene_id, _SceneDiceCounts.zeros(self.num_classes)
            )
            sample_valid = valid[sample_index]
            sample_pred = pred[sample_index]
            sample_target = target[sample_index]
            sample_counts = self._counts_for_mask(
                sample_pred, sample_target, sample_valid
            )
            scene.intersection += sample_counts[0]
            scene.predicted += sample_counts[1]
            scene.target += sample_counts[2]
            pred_fg = (sample_pred > 0) & sample_valid
            target_fg = (sample_target > 0) & sample_valid
            scene.binary_intersection += float((pred_fg & target_fg).sum().item())
            scene.binary_predicted += float(pred_fg.sum().item())
            scene.binary_target += float(target_fg.sum().item())

    def _payload(self) -> dict[str, Any]:
        return {
            "global_intersection": self.global_intersection,
            "global_predicted": self.global_predicted,
            "global_target": self.global_target,
            "batch_dice_sum": self.batch_dice_sum,
            "batch_iou_sum": self.batch_iou_sum,
            "batch_fg_dice_sum": self.batch_fg_dice_sum,
            "batch_fg_iou_sum": self.batch_fg_iou_sum,
            "batch_fg_sample_count": self.batch_fg_sample_count,
            "batch_hd95_sum": self.batch_hd95_sum,
            "batch_sample_count": self.batch_sample_count,
            "scene_counts": self.scene_counts,
        }

    def _reset_statistics(self) -> None:
        self.global_intersection.fill(0.0)
        self.global_predicted.fill(0.0)
        self.global_target.fill(0.0)
        self.scene_counts.clear()
        self.batch_dice_sum = 0.0
        self.batch_iou_sum = 0.0
        self.batch_fg_dice_sum = 0.0
        self.batch_fg_iou_sum = 0.0
        self.batch_fg_sample_count = 0
        self.batch_hd95_sum = 0.0
        self.batch_sample_count = 0

    def _merge_payload(self, payload: dict[str, Any]) -> None:
        self.global_intersection += payload["global_intersection"]
        self.global_predicted += payload["global_predicted"]
        self.global_target += payload["global_target"]
        self.batch_dice_sum += float(payload["batch_dice_sum"])
        self.batch_iou_sum += float(payload["batch_iou_sum"])
        self.batch_fg_dice_sum += float(payload["batch_fg_dice_sum"])
        self.batch_fg_iou_sum += float(payload["batch_fg_iou_sum"])
        self.batch_fg_sample_count += int(payload["batch_fg_sample_count"])
        self.batch_hd95_sum += float(payload["batch_hd95_sum"])
        self.batch_sample_count += int(payload["batch_sample_count"])
        for scene_id, other_scene in payload["scene_counts"].items():
            scene = self.scene_counts.setdefault(
                scene_id, _SceneDiceCounts.zeros(self.num_classes)
            )
            scene.merge_(other_scene)

    def synchronize_between_processes(self) -> None:
        """Merge raw metric state across DDP ranks before computing ratios."""
        if not dist.is_available() or not dist.is_initialized():
            return
        payloads: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(payloads, self._payload())
        self._reset_statistics()
        for payload in payloads:
            if payload is not None:
                self._merge_payload(payload)

    def compute(self) -> dict[str, Any]:
        dice_results = {
            name: DICE_METRIC_FUNCTIONS[name](self) for name in self.metric_names
        }
        iou_results = {
            name: IOU_METRIC_FUNCTIONS[name](self) for name in IOU_METRIC_NAMES
        }
        denominator = max(self.batch_sample_count, 1)
        return {
            "Dice": dice_results,
            # Keep this historical scalar intact for old parsers/checkpoints.
            "IoU": float(self.batch_iou_sum / denominator),
            "IoU_metrics": iou_results,
            "HD95": float(self.batch_hd95_sum / denominator),
        }


class AverageMeter:
    """与 reference_code/lotsnet/utils/losses.py 一致的滑动平均计数器。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def compute_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_weights: list[float] | None = None,
    hd95_backend: str = "scipy",
    dice_metrics: Sequence[str] | str | None = None,
    scene_ids: Sequence[str] | None = None,
    ignore_index: int = -1,
) -> dict[str, Any]:
    """Compute selected Dice protocols, all IoU protocols, and HD95.

    This one-shot entry point treats its input as the complete evaluation scope.
    Training/validation code that spans multiple mini-batches must instead call
    :class:`SegmentationMetricAccumulator.update` for every batch and call
    ``compute`` only after the complete loader has been consumed.

    hd95_backend:
        - "scipy": CPU 边界(surface)距离变换，训练更稳定（默认）。
          算法已与 monai 的 percentile/方向定义对齐，数值与 "monai" 后端等效
          （详见 ``hd95_score`` docstring），仅计算设备不同。
        - "monai": 与 LoTS-Net reference 一致的 MONAI 实现（GPU）
    """
    accumulator = SegmentationMetricAccumulator(
        num_classes=num_classes,
        dice_metrics=dice_metrics,
        class_weights=class_weights,
        hd95_backend=hd95_backend,
        ignore_index=ignore_index,
    )
    accumulator.update(logits, target, scene_ids=scene_ids)
    return accumulator.compute()


def compute_hd95(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    backend: str = "scipy",
    ignore_index: int = -1,
) -> float:
    """按 backend 计算 HD95。"""
    if backend == "monai":
        return monai_hd95_score(pred, target, num_classes, ignore_index=ignore_index)
    if backend == "scipy":
        return hd95_score(pred, target, num_classes, ignore_index=ignore_index)
    raise ValueError(
        f"backend must be one of {HD95_BACKEND_CHOICES}, got {backend!r}"
    )


def monai_hd95_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """与 reference_code/lotsnet/utils/losses.py::compute_metrics 中 MONAI 路径一致。"""
    if not _MONAI_AVAILABLE:
        raise ImportError(
            "hd95_backend='monai' requires monai; install monai or use hd95_backend='scipy'"
        )

    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    y_pred_onehot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
    y_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if ignore_index >= 0:
        keep = [c for c in range(num_classes) if c != ignore_index]
        y_pred_onehot = y_pred_onehot[:, keep]
        y_onehot = y_onehot[:, keep]

    try:
        hd95_per_class_per_batch = compute_hausdorff_distance(
            y_pred_onehot,
            y_onehot,
            percentile=95,
            include_background=False,
        )
        hd95_per_class_per_batch[torch.isinf(hd95_per_class_per_batch)] = torch.nan
        return torch.nanmean(hd95_per_class_per_batch).item()
    except Exception as exc:  # noqa: BLE001 - 与 reference 行为一致，兜底不中断训练
        print(exc)
        return 100.0


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = -1,
) -> torch.Tensor:
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    dice_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = (pred == c).float()
        tgt_c = (target == c).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum()
        dice_per_class.append((2.0 * inter + smooth) / (union + smooth))
    return torch.stack(dice_per_class).mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = -1,
) -> torch.Tensor:
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    iou_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = (pred == c).float()
        tgt_c = (target == c).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum() - inter
        iou_per_class.append((inter + smooth) / (union + smooth))
    return torch.stack(iou_per_class).mean()


def batch_fg_macro_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[float, float]:
    """Return one batch's foreground macro Dice and IoU.

    The batch is pooled spatially before scoring. Background is excluded;
    classes absent from both prediction and target are skipped, while one-sided
    classes contribute zero. When no foreground class is evaluable, both
    results are NaN so callers can skip that batch.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    valid = target != ignore_index
    dice_values: list[float] = []
    iou_values: list[float] = []
    for class_index in range(1, int(num_classes)):
        pred_c = (pred == class_index) & valid
        target_c = (target == class_index) & valid
        predicted = int(pred_c.sum().item())
        target_area = int(target_c.sum().item())
        if predicted + target_area <= 0:
            continue
        intersection = int((pred_c & target_c).sum().item())
        dice_values.append(2.0 * intersection / (predicted + target_area))
        iou_values.append(
            intersection / (predicted + target_area - intersection)
        )
    if not dice_values:
        return float("nan"), float("nan")
    return float(np.mean(dice_values)), float(np.mean(iou_values))


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    return (pred == target).float().mean()


def hd95_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    include_background: bool = False,
    percentile: float = 95.0,
) -> float:
    """基于边界(surface)距离的 95% 分位数 Hausdorff 距离，纯 CPU/scipy 实现。

    与 ``monai.metrics.compute_hausdorff_distance(percentile=95, include_background=False)``
    数值等效（已用随机形状数值验证，误差在 float32 精度范围内）。核心对齐点：
        1. 只用二值腐蚀 XOR (``binary_erosion(mask) ^ mask``) 提取的边界像素参与统计，
           而不是整片填充区域——填充区域里重叠部分距离恒为 0，会把统计量严重稀释，
           这也是旧版 scipy 实现数值远小于 MONAI 的根本原因；
        2. pred->gt 与 gt->pred 两个方向分别取 95% 分位数，再取两者的较大值
           （而不是把两个方向的距离拼接成一个数组后统一取一次分位数）。

    与 MONAI 的一点主动偏差（为训练稳定性保留，非 bug）：
        MONAI 在某一侧掩膜为空时把该类别记为 NaN，最终靠 ``nanmean`` 跳过；这里沿用旧版
        约定——两侧都为空记 0（预测和真值都认为没有该类别，完全一致），只有一侧为空记为
        图像对角线长度作为惩罚值。这样可以避免训练早期模型还学不会预测前景时，NaN 顺着
        ``dist.all_reduce`` 污染整个 epoch 的验证指标，同时不影响“正常有重叠可比较”场景下
        与 MONAI 的数值等效性。
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = pred_np[None]
        tgt_np = tgt_np[None]

    h, w = pred_np.shape[-2], pred_np.shape[-1]
    diag = float((h ** 2 + w ** 2) ** 0.5)

    def _edges(mask: np.ndarray) -> np.ndarray:
        return binary_erosion(mask) ^ mask

    def _surface_distance(edges_a: np.ndarray, edges_b: np.ndarray) -> np.ndarray:
        if not edges_b.any():
            return np.full(int(edges_a.sum()), np.inf)
        dt = distance_transform_edt(~edges_b)
        return dt[edges_a]

    def _single(pred_c: np.ndarray, tgt_c: np.ndarray) -> float:
        if not pred_c.any() and not tgt_c.any():
            return 0.0
        if not pred_c.any() or not tgt_c.any():
            return diag
        edges_pred, edges_gt = _edges(pred_c), _edges(tgt_c)
        d_pred_to_gt = _surface_distance(edges_pred, edges_gt)
        d_gt_to_pred = _surface_distance(edges_gt, edges_pred)
        p1 = float(np.percentile(d_pred_to_gt, percentile)) if d_pred_to_gt.size else diag
        p2 = float(np.percentile(d_gt_to_pred, percentile)) if d_gt_to_pred.size else diag
        return max(p1, p2)

    start_c = 0 if include_background else 1
    hd_vals: list[float] = []
    for c in range(start_c, num_classes):
        if c == ignore_index:
            continue
        per_sample = [
            _single(pred_np[b] == c, tgt_np[b] == c) for b in range(pred_np.shape[0])
        ]
        hd_vals.append(float(np.mean(per_sample)))

    return float(sum(hd_vals) / len(hd_vals)) if hd_vals else float("nan")


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    m_pix: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """m_pix: 1=可见, 0=loss 区域；PSNR 在 loss 区域计算。"""
    inv_m = (1.0 - m_pix).clamp(min=0.0)
    if inv_m.dim() == 4 and pred.dim() == 4:
        inv_m = inv_m.expand_as(pred)
    n = inv_m.sum().clamp(min=1.0)
    mse = (inv_m * (pred - target).pow(2)).sum() / n
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10.0 * torch.log10(data_range ** 2 / mse)
