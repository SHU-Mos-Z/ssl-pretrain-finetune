"""
SpectralAggregator：输入波段数完全自适应的空间特征提取器。

将任意 C 波段输入拆分为 C 个单波段，使用共享权重 Conv2d(1→out_ch) 提取空间特征，
再通过 mean / max / attention 三种方式跨波段聚合。
入口卷积权重形状固定为 (out_ch, 1, kH, kW)，与 C 完全无关。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODES = frozenset({"mean", "max", "attention"})


class SpectralAggregator(nn.Module):
    """
    (B, C, H, W)
      → reshape (B·C, 1, H, W)
      → shared Conv2d(1, out_ch)
      → (B, C, out_ch, H', W')
      → 聚合 dim=1
      → (B, out_ch, H', W')

    Args:
        out_ch      : 输出通道数。
        kernel_size : 共享卷积核大小（默认 3）。
        stride      : 卷积步长（patch embedding 时设为 patch_size）。
        padding     : 卷积填充。
        mode        : 'mean' | 'max' | 'attention'（默认）。
    """

    def __init__(
        self,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        mode: str = "attention",
    ):
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"mode 须为 {sorted(_MODES)}，收到 '{mode}'")
        self.mode = mode
        self.out_ch = out_ch
        self.shared_conv = nn.Conv2d(
            1, out_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        if mode == "attention":
            self.attn_fc = nn.Conv2d(out_ch, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.attn_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.shared_conv(x.reshape(B * C, 1, H, W))  # (B·C, out_ch, H', W')
        _, _, Hf, Wf = feat.shape
        feat = feat.reshape(B, C, self.out_ch, Hf, Wf)

        if self.mode == "mean":
            return feat.mean(dim=1)
        if self.mode == "max":
            return feat.max(dim=1).values

        # attention
        attn = self.attn_fc(feat.reshape(B * C, self.out_ch, Hf, Wf))  # (B·C,1,H',W')
        attn = F.softmax(attn.reshape(B, C, 1, Hf, Wf), dim=1)
        return (feat * attn).sum(dim=1)
