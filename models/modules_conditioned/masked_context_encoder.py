"""CNN context path using physical filling and explicit visibility evidence."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules_cinet.contextual_encoder import ContextualEncoder


class MaskedContextEncoder(nn.Module):
    def __init__(self, stem_ch: int, spatial_depth: int, spectral_agg: str):
        super().__init__()
        self.encoder = ContextualEncoder(stem_ch, spatial_depth, spectral_agg)
        self.visibility_stem = nn.Sequential(
            nn.Conv2d(2, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = self.encoder.out_ch
        self.layer_channels = self.encoder.layer_channels
        self.stem_ch = stem_ch

    def forward(
        self,
        od: torch.Tensor,
        x0: torch.Tensor,
        voxel_visible: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        visible = voxel_visible.to(dtype=od.dtype)
        od_filled = visible * od + (1.0 - visible) * x0.detach()
        mask_features = self.visibility_stem(
            torch.cat((visible.mean(dim=1, keepdim=True), rho), dim=1)
        )
        x = self.encoder.stem(od_filled) + mask_features
        skips = [x]
        for layer in self.encoder.layers:
            x = layer(x)
            skips.append(x)
        return x, skips
