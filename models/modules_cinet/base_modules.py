"""CNN 基础构件：ConvBnRelu / StandardResidualBlock / DownsampleResidualBlock / ResidualBottleneck。"""

import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class StandardResidualBlock(nn.Module):
    """SRB: 1×1 → 3×3(stride=1) → 1×1，空间尺寸不变。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(out_ch // 4, 1)
        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, padding=1),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class DownsampleResidualBlock(nn.Module):
    """DRB: 1×1 → 3×3(stride=2) → 1×1，空间尺寸减半、通道变换。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(out_ch // 4, 1)
        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class ResidualBottleneck(nn.Module):
    """RB = DRB + 4×SRB（下采样 ×2，通道数变换）。"""

    def __init__(self, in_ch: int, out_ch: int, num_srb: int = 4):
        super().__init__()
        self.drb = DownsampleResidualBlock(in_ch, out_ch)
        self.srbs = nn.Sequential(
            *[StandardResidualBlock(out_ch, out_ch) for _ in range(num_srb)]
        )

    def forward(self, x):
        return self.srbs(self.drb(x))
