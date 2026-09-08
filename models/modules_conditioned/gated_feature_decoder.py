"""Spatial/channel gated decoder producing the downstream feature map."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.x_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.spatial_gate = nn.Sequential(nn.Conv2d(2 * out_ch + 1, 1, 3, padding=1), nn.Sigmoid())
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * out_ch, max(out_ch // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_ch // 4, 1), out_ch, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        rho = F.interpolate(rho, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x_p, s_p = self.x_proj(x), self.skip_proj(skip)
        spatial = self.spatial_gate(torch.cat((x_p, s_p, rho), dim=1))
        channel = self.channel_gate(torch.cat((x_p, s_p), dim=1))
        return self.refine(x_p + spatial * channel * s_p)


class GatedFeatureDecoder(nn.Module):
    def __init__(
        self,
        vit_dim: int,
        stem_ch: int,
        layer_channels: list[int],
        mid_ch: int,
        feature_dim: int,
    ):
        super().__init__()
        skip_channels = list(reversed(layer_channels)) + [stem_ch]
        blocks = []
        in_ch = vit_dim
        for skip_ch in skip_channels:
            blocks.append(_GatedFuse(in_ch, skip_ch, mid_ch))
            in_ch = mid_ch
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(mid_ch, feature_dim, 1)

    def forward(
        self,
        f_low: torch.Tensor,
        skips: list[torch.Tensor],
        rho: torch.Tensor,
        return_intermediate: bool = False,
        stop_at_stage: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if stop_at_stage < 0 or stop_at_stage >= len(self.blocks):
            raise ValueError(f"stop_at_stage must be in [0,{len(self.blocks) - 1}]")
        if stop_at_stage and not return_intermediate:
            raise ValueError("early decoder stopping requires return_intermediate=True")
        x = f_low
        intermediate: dict[str, torch.Tensor] = {}
        highest_stage = len(self.blocks) - 1
        for index, (block, skip) in enumerate(zip(self.blocks, reversed(skips))):
            x = block(x, skip, rho)
            stage = highest_stage - index
            intermediate[f"D{stage}"] = x
            if stage == stop_at_stage:
                break
        output = self.output(x) if stop_at_stage == 0 else x
        if return_intermediate:
            return output, intermediate
        return output
