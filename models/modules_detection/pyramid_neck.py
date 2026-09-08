"""Detection feature adapters for the three planned feature modes."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        # Keep at least two channels per group so a B=1, H=W=1 pyramid
        # level remains valid during training.
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )


class ZFullNeck(nn.Module):
    """Project the full-resolution decoder output without spatial resampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = ConvNormAct(in_channels, out_channels, 1)

    def forward(self, z: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(P0=self.projection(z))


class ZPyramidNeck(nn.Module):
    """Build P2-P5 from final Z by learned stride-2 convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.to_half = ConvNormAct(in_channels, out_channels, 3, stride=2)
        self.to_p2 = ConvNormAct(out_channels, out_channels, 3, stride=2)
        self.to_p3 = ConvNormAct(out_channels, out_channels, 3, stride=2)
        self.to_p4 = ConvNormAct(out_channels, out_channels, 3, stride=2)
        self.to_p5 = ConvNormAct(out_channels, out_channels, 3, stride=2)

    def forward(self, z: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.to_half(z)
        p2 = self.to_p2(x)
        p3 = self.to_p3(p2)
        p4 = self.to_p4(p3)
        p5 = self.to_p5(p4)
        return OrderedDict(P2=p2, P3=p3, P4=p4, P5=p5)


class GatedPyramidNeck(nn.Module):
    """Project native gated-decoder D2-D4 stages and derive P5."""

    def __init__(self, decoder_channels: int, out_channels: int):
        super().__init__()
        self.p2 = ConvNormAct(decoder_channels, out_channels, 1)
        self.p3 = ConvNormAct(decoder_channels, out_channels, 1)
        self.p4 = ConvNormAct(decoder_channels, out_channels, 1)
        self.p5 = ConvNormAct(out_channels, out_channels, 3, stride=2)

    def forward(
        self, decoder_stages: dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        missing = {"D2", "D3", "D4"}.difference(decoder_stages)
        if missing:
            raise KeyError(f"gated decoder stages missing: {sorted(missing)}")
        p2 = self.p2(decoder_stages["D2"])
        p3 = self.p3(decoder_stages["D3"])
        p4 = self.p4(decoder_stages["D4"])
        return OrderedDict(P2=p2, P3=p3, P4=p4, P5=self.p5(p4))
