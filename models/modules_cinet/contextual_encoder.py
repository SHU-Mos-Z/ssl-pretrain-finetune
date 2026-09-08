"""
ContextualEncoder：全卷积上下文编码器。

接受（可选空间遮蔽的）OD 数据立方 (B, S, H, W)，
通过 SpectralAggregator Stem + 逐层 ResidualBottleneck 生成多尺度特征图。

每层 ResidualBottleneck 将空间分辨率下采样 ×2，
共 spatial_depth 层，总下采样率 patch_size = 2 ** spatial_depth。
须与 ViT 路径的 patch_size 保持一致。

输出：
  e_c   : (B, layer_channels[-1], H_p, W_p)  最终特征图
  skips : list[(B, ch, H/2^i, W/2^i)]         多尺度跳跃连接
            skips[0] = stem 输出  (B, stem_ch, H,   W  )
            skips[1] = RB1 输出  (B, 128,     H/2, W/2)
            ...
            skips[-1] = RB_last (B, 256,     H_p, W_p)
"""

import math
import torch.nn as nn
from models.modules_cinet.base_modules import ResidualBottleneck
from models.modules_cinet.spectral_aggregator import SpectralAggregator

_SPECTRAL_AGG_MODES = frozenset({"mean", "max", "attention"})


class ContextualEncoder(nn.Module):
    """
    Args:
        stem_ch      : Stem 输出通道，默认 64。
        spatial_depth: 下采样层数，patch_size = 2**spatial_depth。
                       须 >= 1，须与 ViT 路径的 patch_size = 2**spatial_depth 一致。
        spectral_agg : SpectralAggregator 模式（'mean'/'max'/'attention'）。
    """

    def __init__(
        self,
        stem_ch: int = 64,
        spatial_depth: int = 4,
        spectral_agg: str = "attention",
    ):
        super().__init__()
        if spatial_depth < 1:
            raise ValueError(f"spatial_depth >= 1，收到 {spatial_depth}")
        if spectral_agg not in _SPECTRAL_AGG_MODES:
            raise ValueError(f"spectral_agg 须为 {sorted(_SPECTRAL_AGG_MODES)}，收到 '{spectral_agg}'")

        self.patch_size = 2 ** spatial_depth

        # 通道列表：[128, 256, 256, ...]，长度 = spatial_depth
        layer_channels = [128] + [256] * (spatial_depth - 1)
        self.layer_channels = layer_channels

        self.stem = nn.Sequential(
            SpectralAggregator(stem_ch, kernel_size=3, stride=1, padding=1, mode=spectral_agg),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.stem_ch = stem_ch

        self.layers = nn.ModuleList()
        in_ch = stem_ch
        for out_ch in layer_channels:
            self.layers.append(ResidualBottleneck(in_ch, out_ch))
            in_ch = out_ch

        self.out_ch = layer_channels[-1]  # 供外部查询

    def forward(self, x):
        """
        Args:
            x: (B, S, H, W)  原始或空间遮蔽后的 OD 数据立方
        Returns:
            e_c  : (B, out_ch, H_p, W_p)
            skips: list of feature maps，从 stem 到最后一层
        """
        skips = []
        x = self.stem(x)
        skips.append(x)
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        return x, skips
