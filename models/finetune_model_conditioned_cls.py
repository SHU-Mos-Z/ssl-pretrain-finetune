"""Patch-level classification model for the conditioned HSI backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.conditioned_contracts import ConditionedModelConfig
from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel


CLASSIFICATION_HEAD_TYPES = (
    "h0_gap_linear",
    "h1_attention_mlp",
    "h2_dual_scale",
    "h3_multiscale_gated",
)


def _require_feature(
    output: dict[str, Any], key: str, *, channels_last: bool = False
) -> torch.Tensor:
    feature = output.get(key)
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise ValueError(f"backbone output '{key}' must have shape (B,C,H,W)")
    if channels_last:
        feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature


class _ProjectedAttentionPool(nn.Module):
    """Project a feature level and learn a normalized spatial pooling map."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # Classification uses small batches, so avoid batch-dependent statistics.
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.attention = nn.Conv2d(out_channels, 1, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("classification features must have shape (B,C,H,W)")
        projected = self.projection(feature)
        values = projected.flatten(2)
        weights = self.attention(projected).flatten(2).softmax(dim=-1)
        return torch.bmm(values, weights.transpose(1, 2)).squeeze(-1)


class _MlpClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.layers(self.norm(vector))


class ConditionedClassificationHead(nn.Module):
    """H0: original global-average-pooling linear baseline."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, output: dict[str, Any] | torch.Tensor) -> torch.Tensor:
        # Accepting a tensor preserves the public behavior of the original H0 head.
        features = (
            output
            if isinstance(output, torch.Tensor)
            else _require_feature(output, "features")
        )
        if features.ndim != 4:
            raise ValueError("classification features must have shape (B,D,H,W)")
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(self.norm(pooled)))


class AttentionMlpClassificationHead(nn.Module):
    """H1: attention-pool the final decoder feature Z, then use a small MLP."""

    def __init__(
        self,
        in_channels: int,
        projection_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.pool = _ProjectedAttentionPool(in_channels, projection_dim)
        self.classifier = _MlpClassifier(
            projection_dim, hidden_dim, num_classes, dropout
        )

    def forward(self, output: dict[str, Any]) -> torch.Tensor:
        return self.classifier(self.pool(_require_feature(output, "features")))


class DualScaleClassificationHead(nn.Module):
    """H2: concatenate attention-pooled f_low and final decoder feature Z."""

    def __init__(
        self,
        low_channels: int,
        final_channels: int,
        projection_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.low_pool = _ProjectedAttentionPool(low_channels, projection_dim)
        self.final_pool = _ProjectedAttentionPool(final_channels, projection_dim)
        self.classifier = _MlpClassifier(
            2 * projection_dim, hidden_dim, num_classes, dropout
        )

    def forward(self, output: dict[str, Any]) -> torch.Tensor:
        low = self.low_pool(_require_feature(output, "f_low", channels_last=True))
        final = self.final_pool(_require_feature(output, "features"))
        return self.classifier(torch.cat((low, final), dim=1))


class MultiscaleGatedClassificationHead(nn.Module):
    """H3: attention-pool f_low/D2/Z and fuse them with sample-wise gates."""

    def __init__(
        self,
        low_channels: int,
        decoder_channels: int,
        final_channels: int,
        projection_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.low_pool = _ProjectedAttentionPool(low_channels, projection_dim)
        self.mid_pool = _ProjectedAttentionPool(decoder_channels, projection_dim)
        self.final_pool = _ProjectedAttentionPool(final_channels, projection_dim)
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(3 * projection_dim),
            nn.Linear(3 * projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.classifier = _MlpClassifier(
            projection_dim, hidden_dim, num_classes, dropout
        )

    def forward(self, output: dict[str, Any]) -> torch.Tensor:
        decoder_stages = output.get("decoder_stages")
        if not isinstance(decoder_stages, dict) or "D2" not in decoder_stages:
            raise KeyError("H3 classification requires gated-decoder stage D2")
        descriptors = torch.stack(
            (
                self.low_pool(
                    _require_feature(output, "f_low", channels_last=True)
                ),
                self.mid_pool(decoder_stages["D2"]),
                self.final_pool(_require_feature(output, "features")),
            ),
            dim=1,
        )
        gate_input = descriptors.flatten(1)
        scale_weights = self.scale_gate(gate_input).softmax(dim=1)
        fused = (descriptors * scale_weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused)


def _build_classification_head(
    head_type: str,
    config: ConditionedModelConfig,
    num_classes: int,
    projection_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if head_type == "h0_gap_linear":
        return ConditionedClassificationHead(
            config.feature_dim, num_classes, dropout
        )
    if head_type == "h1_attention_mlp":
        return AttentionMlpClassificationHead(
            config.feature_dim,
            projection_dim,
            hidden_dim,
            num_classes,
            dropout,
        )
    if head_type == "h2_dual_scale":
        return DualScaleClassificationHead(
            config.embed_dim,
            config.feature_dim,
            projection_dim,
            hidden_dim,
            num_classes,
            dropout,
        )
    if head_type == "h3_multiscale_gated":
        if config.patch_size < 4:
            raise ValueError("h3_multiscale_gated requires patch_size>=4 for D2")
        return MultiscaleGatedClassificationHead(
            config.embed_dim,
            config.decoder_mid_ch,
            config.feature_dim,
            projection_dim,
            hidden_dim,
            num_classes,
            dropout,
        )
    raise ValueError(
        f"unknown classification head '{head_type}'; "
        f"expected one of {CLASSIFICATION_HEAD_TYPES}"
    )


class ConditionedClassificationModel(nn.Module):
    """Conditioned backbone plus an independent patch classification head."""

    def __init__(
        self,
        num_classes: int,
        config: ConditionedModelConfig,
        pretrain_ckpt: str | None = None,
        freeze_backbone: bool = False,
        head_dropout: float = 0.1,
        classification_head: str = "h0_gap_linear",
        head_projection_dim: int = 64,
        head_hidden_dim: int = 128,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if head_projection_dim <= 0 or head_hidden_dim <= 0:
            raise ValueError("classification head dimensions must be positive")
        self.config = config
        self.classification_head = classification_head
        self.backbone = EndmemberConditionedPretrainModel(config)
        self.cls_head = _build_classification_head(
            classification_head,
            config,
            num_classes,
            head_projection_dim,
            head_hidden_dim,
            head_dropout,
        )
        self._backbone_frozen = False

        # These pretext heads are not part of classification forward propagation.
        for module in (
            self.backbone.abundance_head,
            self.backbone.token_reconstruction_head,
        ):
            module.requires_grad_(False)

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
        shape_mismatch = 0
        unexpected = 0
        for key, value in state.items():
            normalized = key.removeprefix("module.").removeprefix("backbone.")
            if normalized not in current:
                unexpected += 1
                continue
            if current[normalized].shape != value.shape:
                shape_mismatch += 1
                continue
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
            "[ConditionedClassificationModel] "
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
            # A linear probe must not update BatchNorm running statistics.
            self.backbone.eval()
        return self

    def forward_with_features(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output = self.backbone.forward_features(
            batch,
            return_decoder_stages=(
                self.classification_head == "h3_multiscale_gated"
            ),
        )
        return self.cls_head(output), output

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.forward_with_features(batch)[0]
