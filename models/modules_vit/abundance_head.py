"""丰度头：低分辨率 Conv 预测 + 双线性上采样 + 可选 RefineConv。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceHead(nn.Module):
    """
    (B, H_p, W_p, D) → (B, K, H, W)
    """

    def __init__(
        self,
        embed_dim: int,
        num_endmembers: int = 2,
        upsample_mode: str = "bilinear",
        use_refine: bool = True,
        activation: str = "softmax",
    ):
        super().__init__()
        if activation not in ("softmax", "softplus"):
            raise ValueError(f"activation must be softmax or softplus, got {activation}")
        self.activation = activation
        self.num_endmembers = num_endmembers
        self.upsample_mode = upsample_mode
        self.proj = nn.Conv2d(embed_dim, num_endmembers, kernel_size=1)
        self.refine = None
        if use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(num_endmembers, num_endmembers, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_endmembers),
                nn.ReLU(inplace=True),
            )

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "softmax":
            return F.softmax(x, dim=1)
        return F.softplus(x)

    def forward(self, f: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            f: (B, H_p, W_p, D)
            out_hw: (H, W) 目标空间分辨率
        Returns:
            c_pix: (B, K, H, W)
        """
        h, w = out_hw
        x = f.permute(0, 3, 1, 2).contiguous()        # (B, D, H_p, W_p)
        c_low = self._activate(self.proj(x))           # (B, K, H_p, W_p)
        align = False if self.upsample_mode == "bilinear" else None
        c_up = F.interpolate(
            c_low, size=(h, w), mode=self.upsample_mode,
            align_corners=align,
        )
        if self.refine is not None:
            c_up = self.refine(c_up)
            c_up = self._activate(c_up)
        return c_up


def upsample_features(
    f: torch.Tensor,
    out_hw: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    将低分辨率特征上采样至全图，供分割 backbone 使用。

    Args:
        f: (B, H_p, W_p, D)
    Returns:
        (B, D, H, W)
    """
    h, w = out_hw
    x = f.permute(0, 3, 1, 2).contiguous()
    align = False if mode == "bilinear" else None
    return F.interpolate(x, size=(h, w), mode=mode, align_corners=align)
