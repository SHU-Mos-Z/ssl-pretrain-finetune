"""
PixelDecoder：利用 CNN skip 连接将低分辨率特征逐步上采样至全图分辨率。

输入：
  f_low : (B, H_p, W_p, D)  跨谱聚合后的 ViT 特征（CIAM 增强后）
  skips : list  来自 ContextualEncoder 的多尺度跳跃连接
              skips[0]  = stem 输出  (B, stem_ch, H,   W  )
              skips[1]  = RB1 输出  (B, 128,     H/2, W/2)
              ...
              skips[-1] = RBn 输出  (B, 256,     H_p, W_p)

流程（对称 U-Net 风格）：
  1. 从 skips[-1] 开始（与 f_low 同分辨率），拼接 f_low + skip_last，经 DecoderBlock
  2. 逐层拼接 skip、经 DecoderBlock 上采样，直到与 stem 同分辨率
  3. 拼接 skip_stem，经 final_conv 预测输出通道

out_ch 可配置：
  - 预训练：out_ch = K（端元数），末层接 Softmax/Softplus
  - 微调：  out_ch = num_classes，末层无激活（配合 CrossEntropyLoss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules_cinet.base_modules import ConvBnRelu


class DecoderBlock(nn.Module):
    """Conv×2 + Upsample×2（双线性）。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, out_ch, kernel_size=3, padding=1),
            ConvBnRelu(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.conv(x))


class PixelDecoder(nn.Module):
    """
    Args:
        vit_dim        : ViT 特征维度 D（f_low 的通道数）。
        stem_ch        : ContextualEncoder stem 输出通道（skips[0] 的通道数）。
        layer_channels : ContextualEncoder 各 RB 层通道列表，如 [128, 256, 256, 256]。
        out_ch         : 最终输出通道数（K 或 num_classes）。
        mid_ch         : final_conv 中间通道数，默认 64。
    """

    def __init__(
        self,
        vit_dim: int,
        stem_ch: int,
        layer_channels: list[int],
        out_ch: int,
        mid_ch: int = 64,
    ):
        super().__init__()

        rb_channels_rev = layer_channels[::-1]  # 从最深层到最浅层

        self.blocks = nn.ModuleList()
        current_in = vit_dim  # f_low 初始通道数

        for i, skip_ch in enumerate(rb_channels_rev):
            block_in = current_in + skip_ch
            if i + 1 < len(rb_channels_rev):
                block_out = rb_channels_rev[i + 1]
            else:
                block_out = stem_ch
            self.blocks.append(DecoderBlock(block_in, block_out))
            current_in = block_out

        self.final_conv = nn.Sequential(
            ConvBnRelu(current_in + stem_ch, mid_ch, kernel_size=3, padding=1),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

    def forward(
        self,
        f_low: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            f_low : (B, H_p, W_p, D)
            skips : list from ContextualEncoder
                    skips[0]  = stem  (B, stem_ch, H, W)
                    skips[1:] = RB layers，浅→深顺序
        Returns:
            out: (B, out_ch, H, W)
        """
        skip_stem = skips[0]
        skip_rbs_rev = skips[1:][::-1]  # 深→浅，与 decoder blocks 对应

        assert len(skip_rbs_rev) == len(self.blocks), (
            f"skip_rbs 数量 {len(skip_rbs_rev)} 与 decoder blocks {len(self.blocks)} 不符"
        )

        # (B, H_p, W_p, D) → (B, D, H_p, W_p)
        x = f_low.permute(0, 3, 1, 2).contiguous()

        for block, skip in zip(self.blocks, skip_rbs_rev):
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))

        if x.shape[2:] != skip_stem.shape[2:]:
            x = F.interpolate(x, size=skip_stem.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip_stem], dim=1)

        return self.final_conv(x)
