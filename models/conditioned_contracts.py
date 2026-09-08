"""Shared contracts for the endmember-conditioned pretraining path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


@dataclass(frozen=True)
class ConditionedModelConfig:
    patch_size: int = 16
    spectral_patch_size: int = 5
    embed_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    cnn_stem_ch: int = 64
    cnn_spectral_agg: str = "attention"
    fusion_heads: int = 8
    feature_dim: int = 128
    decoder_mid_ch: int = 64
    residual_hidden_dim: int = 128
    ridge_lambda: float = 1e-3
    confidence_temperature: float = 0.05
    alpha_min: float = 0.1
    alpha_extra: float = 1.0
    od_max: float = 3.0

    def validate(self) -> None:
        if self.patch_size <= 0 or self.patch_size & (self.patch_size - 1):
            raise ValueError("patch_size must be a positive power of two")
        if self.spectral_patch_size <= 0:
            raise ValueError("spectral_patch_size must be positive")
        if self.embed_dim % self.vit_heads:
            raise ValueError("embed_dim must be divisible by vit_heads")
        if self.embed_dim % self.fusion_heads:
            raise ValueError("embed_dim must be divisible by fusion_heads")
        if self.cnn_spectral_agg not in {"mean", "max", "attention"}:
            raise ValueError("invalid cnn_spectral_agg")


class ConditionedBatch(TypedDict):
    od: torch.Tensor
    intensity: torch.Tensor
    e_star: torch.Tensor
    c_star: torch.Tensor
    wavelengths: torch.Tensor
    token_raw: torch.Tensor
    token_visible: torch.Tensor
    voxel_visible: torch.Tensor
    pe_spatial: torch.Tensor
    pe_spectral: torch.Tensor


class ConditionedModelOutput(TypedDict):
    features: torch.Tensor
    c0_low: torch.Tensor
    c0: torch.Tensor
    x0: torch.Tensor
    rho_low: torch.Tensor
    rho: torch.Tensor
    endmember_group_tokens: torch.Tensor
    endmember_tokens: torch.Tensor
    z_vit: torch.Tensor
    z_fused: torch.Tensor
    delta_logits: torch.Tensor
    c_hat: torch.Tensor
    od_hat: torch.Tensor
    i_hat: torch.Tensor
    token_hat: torch.Tensor
