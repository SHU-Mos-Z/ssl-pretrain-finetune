"""Endmember-conditioned backbone with switchable dense detection heads."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.conditioned_contracts import ConditionedModelConfig
from models.detection_contracts import DetectionConfig
from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from models.modules_detection import (
    AnchorGenerator,
    FCOSHead,
    GatedPyramidNeck,
    PointGenerator,
    RetinaNetHead,
    ZFullNeck,
    ZPyramidNeck,
)


class ConditionedDetectionModel(nn.Module):
    """Raw detector forward; assignment, losses and NMS remain external."""

    def __init__(
        self,
        model_config: ConditionedModelConfig,
        detection_config: DetectionConfig,
        pretrain_ckpt: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        model_config.validate()
        detection_config.validate()
        self.model_config = model_config
        self.detection_config = detection_config
        self.backbone = EndmemberConditionedPretrainModel(model_config)
        self._backbone_frozen = False

        # Pretext-only output heads are excluded from detection optimization.
        self.backbone.abundance_head.requires_grad_(False)
        self.backbone.token_reconstruction_head.requires_grad_(False)

        if detection_config.feature_mode == "gated_pyramid" and model_config.patch_size < 16:
            raise ValueError(
                "gated_pyramid requires patch_size>=16 so decoder stages D2-D4 exist"
            )
        if detection_config.feature_mode == "z_full":
            self.neck = ZFullNeck(model_config.feature_dim, detection_config.det_feature_dim)
        elif detection_config.feature_mode == "z_pyramid":
            self.neck = ZPyramidNeck(model_config.feature_dim, detection_config.det_feature_dim)
        else:
            self.neck = GatedPyramidNeck(
                model_config.decoder_mid_ch, detection_config.det_feature_dim
            )
            # gated_pyramid stops after D2; D1, D0 and the final Z projection are
            # intentionally outside this feature path and must not confuse DDP.
            self.backbone.feature_decoder.blocks[-2:].requires_grad_(False)
            self.backbone.feature_decoder.output.requires_grad_(False)

        if detection_config.detection_mode == "anchor_based":
            self.geometry_generator = AnchorGenerator(
                detection_config.anchor_sizes_for_features(),
                detection_config.anchor_scales,
                detection_config.anchor_ratios,
                detection_config.anchor_offset,
            )
            self.head = RetinaNetHead(
                detection_config.det_feature_dim,
                detection_config.num_classes,
                detection_config.anchors_per_location,
                detection_config.head_depth,
                detection_config.prior_probability,
            )
        else:
            self.geometry_generator = PointGenerator(detection_config.anchor_offset)
            self.head = FCOSHead(
                detection_config.det_feature_dim,
                detection_config.num_classes,
                len(detection_config.feature_names),
                detection_config.head_depth,
                detection_config.prior_probability,
            )

        if pretrain_ckpt:
            self.load_pretrain(pretrain_ckpt)
        if freeze_backbone:
            self.freeze_backbone()

    def load_pretrain(self, path: str) -> dict[str, int]:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"pretraining checkpoint does not exist: {path}")
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        if not isinstance(state, dict):
            raise TypeError(f"unsupported checkpoint payload in {path}")
        current = self.backbone.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        unexpected = shape_mismatch = 0
        for key, value in state.items():
            normalized = key.removeprefix("module.").removeprefix("backbone.")
            if normalized not in current:
                unexpected += 1
            elif current[normalized].shape != value.shape:
                shape_mismatch += 1
            else:
                compatible[normalized] = value
        if not compatible:
            raise RuntimeError(f"no compatible backbone weights found in {path}")
        missing, _ = self.backbone.load_state_dict(compatible, strict=False)
        summary = {
            "matched": len(compatible),
            "missing": len(missing),
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
        }
        print(
            "[ConditionedDetectionModel] "
            + " ".join(f"{key}={value}" for key, value in summary.items()),
            flush=True,
        )
        return summary

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self._backbone_frozen = True
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def _feature_maps(self, output: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
        if self.detection_config.feature_mode == "gated_pyramid":
            return self.neck(output["decoder_stages"])
        return self.neck(output["features"])

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        need_stages = self.detection_config.feature_mode == "gated_pyramid"
        backbone_output = self.backbone.forward_features(
            batch,
            return_decoder_stages=need_stages,
            decoder_stop_at_stage=2 if need_stages else 0,
        )
        features = self._feature_maps(backbone_output)
        expected_names = self.detection_config.feature_names
        if tuple(features) != expected_names:
            raise RuntimeError(
                f"detector produced levels {tuple(features)}, expected {expected_names}"
            )
        strides = OrderedDict(zip(expected_names, self.detection_config.feature_strides))
        head_output = self.head(list(features.values()))
        output: dict[str, Any] = {
            **head_output,
            "features": features,
            "feature_names": expected_names,
            "feature_strides": strides,
            "feature_shapes": tuple(tuple(value.shape[-2:]) for value in features.values()),
            "image_size": tuple(batch["od"].shape[-2:]),
            "detection_mode": self.detection_config.detection_mode,
        }
        if self.detection_config.detection_mode == "anchor_based":
            output["anchors"] = self.geometry_generator(features, strides)
        else:
            points, point_strides, level_ids = self.geometry_generator(features, strides)
            output["points"] = points
            output["point_strides"] = point_strides
            output["level_ids"] = level_ids
        return output
