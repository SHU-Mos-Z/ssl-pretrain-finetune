"""
CIAM（Cross-Indication Attention Module）：双向交叉注意力融合模块。

支持 Q/KV 序列长度不等（PyTorch MultiheadAttention 原生支持）：
  - 方向 A（ViT 查询 CNN）：
      Q  = ViT tokens  (B, T_vit, D)，T_vit = H_p·W_p·n_sp
      K,V= CNN tokens  (B, T_cnn, D)，T_cnn = H_p·W_p
      输出 z_from_cnn (B, T_vit, D)

  - 方向 B（CNN 查询 ViT）：
      Q  = CNN tokens  (B, T_cnn, D)
      K,V= ViT tokens  (B, T_vit, D)
      输出 e_from_vit  (B, T_cnn, D)

最终输出取方向 A 的结果（保持 T_vit 个 token 用于后续 Token 一致性头）。
方向 B 的 e_from_vit 可选接入下游（目前作为辅助 skip 返回）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIAM(nn.Module):
    """
    Args:
        embed_dim  : 特征维度 D。
        num_heads  : 多头注意力头数。
        dropout    : Attention dropout。
        ffn_ratio  : FFN 隐藏层倍数（相对 embed_dim）。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 投影层（对齐两路特征维度）
        self.proj_vit = nn.Linear(embed_dim, embed_dim)
        self.proj_cnn = nn.Linear(embed_dim, embed_dim)

        # 方向 A：ViT 查询 CNN
        self.norm_vit_a = nn.LayerNorm(embed_dim)
        self.norm_cnn_a = nn.LayerNorm(embed_dim)
        self.attn_a = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Post-attention FFN for ViT tokens
        self.norm_vit_ffn = nn.LayerNorm(embed_dim)
        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn_vit = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # 方向 B：CNN 查询 ViT
        self.norm_cnn_b = nn.LayerNorm(embed_dim)
        self.norm_vit_b = nn.LayerNorm(embed_dim)
        self.attn_b = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Post-attention FFN for CNN tokens
        self.norm_cnn_ffn = nn.LayerNorm(embed_dim)
        self.ffn_cnn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_vit: torch.Tensor,
        e_cnn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_vit : (B, T_vit, D)  ViT token 序列（已含 PE），T_vit = H_p·W_p·n_sp
            e_cnn : (B, T_cnn, D)  CNN token 序列（展平的空间特征图），T_cnn = H_p·W_p
        Returns:
            z_out    : (B, T_vit, D)  经 CNN 上下文增强的 ViT token（用于后续 Token 一致性头）
            e_out    : (B, T_cnn, D)  经 ViT 全局信息增强的 CNN token（可选用于解码器）
        """
        # ── 投影 ──
        z = self.proj_vit(z_vit)    # (B, T_vit, D)
        e = self.proj_cnn(e_cnn)    # (B, T_cnn, D)

        # ── 方向 A：ViT 查询 CNN（T_vit 个 token 获取 CNN 局部上下文） ──
        q_a = self.norm_vit_a(z)
        kv_a = self.norm_cnn_a(e)
        attn_out_a, _ = self.attn_a(query=q_a, key=kv_a, value=kv_a)  # (B, T_vit, D)
        z = z + attn_out_a                                               # 残差连接
        z = z + self.ffn_vit(self.norm_vit_ffn(z))

        # ── 方向 B：CNN 查询 ViT（T_cnn 个 token 吸收全局谱段上下文） ──
        q_b = self.norm_cnn_b(e)
        kv_b = self.norm_vit_b(z)   # 使用方向 A 更新后的 z
        attn_out_b, _ = self.attn_b(query=q_b, key=kv_b, value=kv_b)  # (B, T_cnn, D)
        e = e + attn_out_b
        e = e + self.ffn_cnn(self.norm_cnn_ffn(e))

        return z, e
