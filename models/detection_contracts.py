"""Configuration shared by conditioned detection model, losses and decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _tuple_floats(values: Any) -> tuple[float, ...]:
    if isinstance(values, str):
        values = values.split(",")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class DetectionConfig:
    detection_mode: str = "anchor_based"
    feature_mode: str = "z_pyramid"
    num_classes: int = 1
    det_feature_dim: int = 128
    head_depth: int = 4
    head_norm: str = "none"
    head_norm_groups: int = 32
    quality_mode: str = "legacy"
    quality_loss_weight: float = 1.0
    quality_score_power: float = 0.5

    anchor_sizes: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)
    anchor_scales: tuple[float, ...] = (1.0, 1.2599, 1.5874)
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    anchor_offset: float = 0.5
    positive_iou_threshold: float = 0.5
    negative_iou_threshold: float = 0.4
    ignore_iou_threshold: float = 0.5
    matcher_chunk_size: int = 65536
    box_coder_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    box_loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0 / 9.0

    fcos_regression_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 32.0),
        (32.0, 64.0),
        (64.0, 128.0),
        (128.0, 1.0e8),
    )
    fcos_center_sampling_radius: float = 1.5
    fcos_normalize_reg_targets_by_stride: bool = True

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss_weight: float = 1.0
    centerness_loss_weight: float = 1.0
    prior_probability: float = 0.01

    score_threshold: float = 0.05
    nms_threshold: float = 0.5
    pre_nms_topk: int = 1000
    max_detections: int = 100
    min_box_size: float = 1.0

    def validate(self) -> None:
        if self.detection_mode not in {"anchor_based", "anchor_free"}:
            raise ValueError("detection_mode must be anchor_based or anchor_free")
        if self.feature_mode not in {"z_pyramid", "gated_pyramid", "gated_fpn", "z_full"}:
            raise ValueError("invalid feature_mode")
        if self.num_classes < 1 or self.det_feature_dim < 1 or self.head_depth < 1:
            raise ValueError("num_classes, det_feature_dim and head_depth must be positive")
        if self.head_norm not in {"none", "group_norm"}:
            raise ValueError("head_norm must be none or group_norm")
        if self.head_norm_groups < 1:
            raise ValueError("head_norm_groups must be positive")
        if self.quality_mode not in {"legacy", "iou"}:
            raise ValueError("quality_mode must be legacy or iou")
        if self.quality_loss_weight < 0:
            raise ValueError("quality_loss_weight must be non-negative")
        if not 0.0 <= self.quality_score_power <= 1.0:
            raise ValueError("quality_score_power must be in [0,1]")
        if not self.anchor_sizes or not self.anchor_scales or not self.anchor_ratios:
            raise ValueError("anchor configuration cannot be empty")
        if any(value <= 0 for value in (*self.anchor_sizes, *self.anchor_scales, *self.anchor_ratios)):
            raise ValueError("anchor sizes/scales/ratios must be positive")
        if not 0 <= self.negative_iou_threshold <= self.positive_iou_threshold <= 1:
            raise ValueError("anchor IoU thresholds are inconsistent")
        if not 0 <= self.ignore_iou_threshold <= 1:
            raise ValueError("ignore_iou_threshold must be in [0,1]")
        if not 0 <= self.score_threshold <= 1 or not 0 <= self.nms_threshold <= 1:
            raise ValueError("score_threshold and nms_threshold must be in [0,1]")
        if not 0 <= self.anchor_offset <= 1:
            raise ValueError("anchor_offset must be in [0,1]")
        if self.matcher_chunk_size < 1:
            raise ValueError("matcher_chunk_size must be positive")
        if self.box_loss not in {"smooth_l1", "giou"}:
            raise ValueError("box_loss must be smooth_l1 or giou")
        if len(self.box_coder_weights) != 4 or any(v <= 0 for v in self.box_coder_weights):
            raise ValueError("box_coder_weights must contain four positive values")
        if self.feature_mode != "z_full" and len(self.fcos_regression_ranges) != 4:
            raise ValueError("P2-P5 require four FCOS regression ranges")
        if self.feature_mode == "z_full" and not self.fcos_regression_ranges:
            raise ValueError("z_full requires at least one FCOS regression range")
        previous = 0.0
        for lower, upper in self.fcos_regression_ranges:
            if lower < 0 or upper <= lower or lower < previous:
                raise ValueError("invalid FCOS regression ranges")
            previous = lower
        if self.feature_mode == "z_full" and len(self.anchor_sizes) < 1:
            raise ValueError("z_full requires at least one anchor size")
        if not 0 < self.prior_probability < 1:
            raise ValueError("prior_probability must be in (0,1)")
        if self.fcos_center_sampling_radius <= 0:
            raise ValueError("fcos_center_sampling_radius must be positive")
        if self.pre_nms_topk < 1 or self.max_detections < 1:
            raise ValueError("postprocessing limits must be positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("P0",) if self.feature_mode == "z_full" else ("P2", "P3", "P4", "P5")

    @property
    def feature_strides(self) -> tuple[int, ...]:
        return (1,) if self.feature_mode == "z_full" else (4, 8, 16, 32)

    @property
    def anchors_per_location(self) -> int:
        return len(self.anchor_scales) * len(self.anchor_ratios)

    def anchor_sizes_for_features(self) -> tuple[float, ...]:
        if self.feature_mode == "z_full":
            return (self.anchor_sizes[0],)
        if len(self.anchor_sizes) != 4:
            raise ValueError("multi-scale detection requires four anchor sizes")
        return self.anchor_sizes

    def fcos_ranges_for_features(self) -> tuple[tuple[float, float], ...]:
        if self.feature_mode == "z_full":
            # A single full-resolution level must cover all trainable object sizes.
            return ((0.0, self.fcos_regression_ranges[-1][1]),)
        return self.fcos_regression_ranges

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionConfig":
        values = dict(payload)
        for key in ("anchor_sizes", "anchor_scales", "anchor_ratios", "box_coder_weights"):
            if key in values:
                values[key] = _tuple_floats(values[key])
        if "fcos_regression_ranges" in values:
            values["fcos_regression_ranges"] = tuple(
                tuple(float(v) for v in pair) for pair in values["fcos_regression_ranges"]
            )
        config = cls(**values)
        config.validate()
        return config
