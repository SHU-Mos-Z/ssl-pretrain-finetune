"""Static candidate/parameter reporting for reproducible detector runs."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.detection_contracts import DetectionConfig


def candidate_statistics(
    model: nn.Module,
    config: DetectionConfig,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    height, width = image_size
    levels = []
    total = 0
    for index, (name, stride) in enumerate(
        zip(config.feature_names, config.feature_strides)
    ):
        feature_height = (height + stride - 1) // stride
        feature_width = (width + stride - 1) // stride
        per_location = config.anchors_per_location if config.detection_mode == "anchor_based" else 1
        candidates = feature_height * feature_width * per_location
        entry: dict[str, Any] = {
            "name": name,
            "stride": stride,
            "feature_shape": [feature_height, feature_width],
            "candidates_per_location": per_location,
            "candidates_per_image": candidates,
        }
        if config.detection_mode == "anchor_based":
            entry.update(
                base_size=config.anchor_sizes_for_features()[index],
                scales=list(config.anchor_scales),
                aspect_ratios=list(config.anchor_ratios),
            )
        else:
            entry.update(
                regression_range=list(config.fcos_ranges_for_features()[index]),
                center_sampling_radius=config.fcos_center_sampling_radius,
            )
        levels.append(entry)
        total += candidates
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "detection_mode": config.detection_mode,
        "feature_mode": config.feature_mode,
        "image_size": [height, width],
        "levels": levels,
        "total_candidates_per_image": total,
        "parameters": all_parameters,
        "trainable_parameters": trainable_parameters,
    }
