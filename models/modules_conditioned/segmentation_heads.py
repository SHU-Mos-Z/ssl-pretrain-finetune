"""Optional segmentation heads for conditioned downstream fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.finetune_model_vit import SegmentationHead


SEGMENTATION_HEAD_TYPES = (
    "h0_simple",
    "h1_residual",
    "h2_aspp",
    "h3_multiscale_aux",
)


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
    ):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = _ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.dropout(self.conv1(x)))
        return self.activation(x + residual)


class ResidualSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = _ConvNormAct(in_channels, hidden_channels)
        self.blocks = nn.Sequential(
            _ResidualBlock(hidden_channels, dropout),
            _ResidualBlock(hidden_channels, dropout),
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.blocks(self.stem(x)))


class ASPPSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        dilation_rates: tuple[int, ...] = (1, 6, 12, 18),
        dropout: float = 0.1,
    ):
        super().__init__()
        if not dilation_rates or any(rate <= 0 for rate in dilation_rates):
            raise ValueError("ASPP dilation rates must be positive")
        self.branches = nn.ModuleList(
            [
                _ConvNormAct(
                    in_channels,
                    hidden_channels,
                    kernel_size=1 if rate == 1 else 3,
                    dilation=1 if rate == 1 else rate,
                )
                for rate in dilation_rates
            ]
        )
        self.project = nn.Sequential(
            _ConvNormAct(hidden_channels * len(dilation_rates), hidden_channels, 1),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.classifier(self.project(features))


class MultiscaleAuxSegmentationHead(nn.Module):
    """Fuse decoder stages D2, D1 and final Z with per-sample scale gates."""

    def __init__(
        self,
        feature_channels: int,
        decoder_channels: int,
        num_classes: int,
        projection_channels: int = 64,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stage_names = ("D2", "D1")
        self.projections = nn.ModuleDict(
            {
                "D2": _ConvNormAct(decoder_channels, projection_channels, 1),
                "D1": _ConvNormAct(decoder_channels, projection_channels, 1),
                "Z": _ConvNormAct(feature_channels, projection_channels, 1),
            }
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(3 * projection_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 3),
        )
        self.refine = nn.Sequential(
            _ConvNormAct(projection_channels, hidden_channels),
            _ResidualBlock(hidden_channels, dropout),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)
        self.auxiliary = nn.ModuleDict(
            {
                name: nn.Conv2d(projection_channels, num_classes, 1)
                for name in self.stage_names
            }
        )

    def forward(
        self, features: torch.Tensor, decoder_stages: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        missing = [name for name in self.stage_names if name not in decoder_stages]
        if missing:
            raise KeyError(f"decoder stages missing for multiscale head: {missing}")
        output_size = features.shape[-2:]
        projected_by_name = {
            name: F.interpolate(
                self.projections[name](decoder_stages[name]),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            for name in self.stage_names
        }
        projected = [projected_by_name[name] for name in self.stage_names]
        projected.append(self.projections["Z"](features))
        descriptors = torch.cat(
            [F.adaptive_avg_pool2d(item, 1).flatten(1) for item in projected], dim=1
        )
        weights = self.scale_gate(descriptors).softmax(dim=1)
        fused = sum(
            item * weights[:, index, None, None, None]
            for index, item in enumerate(projected)
        )
        logits = self.classifier(self.refine(fused))
        auxiliary_logits = [
            self.auxiliary[name](projected_by_name[name])
            for name in self.stage_names
        ]
        return {"logits": logits, "aux_logits": auxiliary_logits, "scale_weights": weights}


def build_segmentation_head(
    head_type: str,
    *,
    feature_channels: int,
    decoder_channels: int,
    num_classes: int,
    hidden_channels: int = 128,
    projection_channels: int = 64,
    dropout: float = 0.1,
    aspp_rates: tuple[int, ...] = (1, 6, 12, 18),
) -> nn.Module:
    if head_type == "h0_simple":
        # Reuse the historical class verbatim so default state_dict keys and
        # numerical behavior remain unchanged.
        return SegmentationHead(feature_channels, num_classes)
    if head_type == "h1_residual":
        return ResidualSegmentationHead(
            feature_channels, num_classes, hidden_channels, dropout
        )
    if head_type == "h2_aspp":
        return ASPPSegmentationHead(
            feature_channels, num_classes, projection_channels, aspp_rates, dropout
        )
    if head_type == "h3_multiscale_aux":
        return MultiscaleAuxSegmentationHead(
            feature_channels,
            decoder_channels,
            num_classes,
            projection_channels,
            hidden_channels,
            dropout,
        )
    raise ValueError(f"head_type must be one of {SEGMENTATION_HEAD_TYPES}")


__all__ = [
    "SEGMENTATION_HEAD_TYPES",
    "ResidualSegmentationHead",
    "ASPPSegmentationHead",
    "MultiscaleAuxSegmentationHead",
    "build_segmentation_head",
]
