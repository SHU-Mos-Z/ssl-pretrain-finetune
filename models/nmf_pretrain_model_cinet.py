"""
NMF 前置 + 规则 Patch × 谱段组 CINET 预训练模型。

前向流程：
  1. 空间遮蔽 OD 立方 → ContextualEncoder(CNN) → e_c + skips
  2. TokenEncoder(3D tokenize + mask) → ViT → z_seq
  3. CIAM(z_vit, e_cnn) → z_out (ViT tokens 经 CNN 上下文增强)
  4. reshape z_out → z_grid (B, H_p, W_p, n_sp, D)
  5. TokenConsistencyHead(z_grid)        → c_tok   [Loss 4]
  6. SpectralAggregate(z_grid)           → f_low
  7. PixelDecoder(f_low, skips)          → c_pix   [Loss 3]
  8. reconstruct_od / reconstruct_intensity         [Loss 1, 2]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.modules_vit.physics_decode import reconstruct_intensity, reconstruct_od
from models.modules_vit.spectral_aggregate import SpectralAggregate
from models.modules_vit.token_consistency_head import TokenConsistencyHead
from models.modules_vit.token_encoder import TokenEncoder
from models.modules_vit.vit_backbone import ViTBackbone
from models.modules_cinet.ciam import CIAM
from models.modules_cinet.contextual_encoder import ContextualEncoder
from models.modules_cinet.pixel_decoder import PixelDecoder
from models.nmf_pretrain_model_vit import seq_to_grid


def _build_spatial_mask(
    is_masked: torch.Tensor,
    h_p: int,
    w_p: int,
    n_sp: int,
    patch_size: int,
) -> torch.Tensor:
    """
    由 Token 级 is_masked (B, T) 生成像素级空间掩膜 (B, 1, H, W)。

    策略（方案 Y）：若空间 Patch (u,v) 中任意谱段组被 mask，
    则该 Patch 所有像素对 CNN 完全遮蔽（所有波段归零），
    防止 CNN 路径出现信息泄露。

    Returns:
        mask_pix : (B, 1, H, W)，1 = 可见，0 = 遮蔽
    """
    B = is_masked.shape[0]
    # (B, T) → (B, H_p, W_p, n_sp)
    is_masked_grid = is_masked.view(B, h_p, w_p, n_sp)
    # 任意谱段被 mask → 该 Patch 视为 mask
    spatial_mask = (~is_masked_grid.any(dim=-1)).float()       # (B, H_p, W_p)，1=可见
    # 用 repeat_interleave 将 patch 级 mask 精确扩展至像素级
    spatial_mask = (
        spatial_mask
        .unsqueeze(1)                                          # (B,1,H_p,W_p)
        .repeat_interleave(patch_size, dim=2)                  # (B,1,H,W_p)
        .repeat_interleave(patch_size, dim=3)                  # (B,1,H,W)
    )
    return spatial_mask


class NMFPretrainModelCINET(nn.Module):
    """
    CINET 版预训练模型（CNN 上下文编码器 + ViT 谱段 Token 编码器 + CIAM 交叉注意力）。

    超参数与纯 ViT 版（NMFPretrainModel）保持兼容，额外增加 CNN 路径配置。
    """

    def __init__(
        self,
        # ── ViT / Token 路径（与纯 ViT 版一致） ──
        embed_dim: int = 256,
        spectral_patch_size: int = 10,
        num_endmembers: int = 2,
        num_spectral_groups: int = 6,
        patch_size: int = 16,
        vit_depth: int = 6,
        vit_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        aggregate_mode: str = "mean",
        abundance_activation: str = "softmax",
        od_max: float = 3.0,
        # ── CNN 路径 ──
        cnn_stem_ch: int = 64,
        cnn_spectral_agg: str = "attention",
        # ── CIAM ──
        ciam_heads: int = 8,
        ciam_dropout: float = 0.1,
        ciam_ffn_ratio: float = 2.0,
        # ── Decoder ──
        decoder_mid_ch: int = 64,
    ):
        super().__init__()
        import math
        spatial_depth = int(math.log2(patch_size))
        assert 2 ** spatial_depth == patch_size, (
            f"patch_size={patch_size} 须为 2 的整数次幂"
        )

        self.embed_dim = embed_dim
        self.spectral_patch_size = spectral_patch_size
        self.num_endmembers = num_endmembers
        self.num_spectral_groups = num_spectral_groups
        self.patch_size = patch_size
        self.od_max = od_max

        # ── ViT 路径 ──
        self.token_encoder = TokenEncoder(
            embed_dim=embed_dim,
            spectral_patch_size=spectral_patch_size,
            patch_size=patch_size,
        )
        self.vit = ViTBackbone(
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ── CNN 路径 ──
        self.cnn_encoder = ContextualEncoder(
            stem_ch=cnn_stem_ch,
            spatial_depth=spatial_depth,
            spectral_agg=cnn_spectral_agg,
        )

        # ── CIAM ──
        self.ciam = CIAM(
            embed_dim=embed_dim,
            num_heads=ciam_heads,
            dropout=ciam_dropout,
            ffn_ratio=ciam_ffn_ratio,
        )
        # CNN 特征维度 → embed_dim 对齐投影
        self.cnn_proj = nn.Linear(self.cnn_encoder.out_ch, embed_dim)

        # ── 预测头 ──
        self.token_cons_head = TokenConsistencyHead(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_endmembers=num_endmembers,
        )
        self.spectral_aggregate = SpectralAggregate(embed_dim, mode=aggregate_mode)

        # ── PixelDecoder（代替纯双线性 AbundanceHead） ──
        self.pixel_decoder = PixelDecoder(
            vit_dim=embed_dim,
            stem_ch=cnn_stem_ch,
            layer_channels=self.cnn_encoder.layer_channels,
            out_ch=num_endmembers,
            mid_ch=decoder_mid_ch,
        )
        self._abundance_activation = abundance_activation

    def _activate_abundance(self, x: torch.Tensor) -> torch.Tensor:
        if self._abundance_activation == "softmax":
            return F.softmax(x, dim=1)
        return F.softplus(x)

    @staticmethod
    def grid_size(h: int, w: int, patch_size: int) -> tuple[int, int]:
        assert h % patch_size == 0 and w % patch_size == 0
        return h // patch_size, w // patch_size

    def forward(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        batch 必含字段（与纯 ViT 版一致，额外需要 'od' 用于 CNN 遮蔽输入）：
            token_raw     (B, T, P*P*s_p)
            is_masked     (B, T) bool
            pe_spatial    (B, T, 2)
            pe_spectral   (B, T)
            od            (B, S, H, W)  原始 OD 数据立方
            e_star        (K, S)
            c_star_patch  (B, H_p, W_p, P*P*K)
            H, W          int
        """
        token_raw = batch["token_raw"]
        is_masked = batch["is_masked"]
        pe_spatial = batch["pe_spatial"]
        pe_spectral = batch["pe_spectral"]
        e_star = batch["e_star"]
        od = batch["od"]
        h, w = int(batch["H"]), int(batch["W"])

        h_p, w_p = self.grid_size(h, w, self.patch_size)
        n_sp = self.num_spectral_groups

        # ── Step A：CNN 路径（方案 Y 空间遮蔽） ──
        mask_pix = _build_spatial_mask(is_masked, h_p, w_p, n_sp, self.patch_size)
        od_masked = od * mask_pix                            # (B, S, H, W)
        e_c, skips = self.cnn_encoder(od_masked)             # e_c: (B, cnn_out_ch, H_p, W_p)

        # CNN 特征展平为 token 序列并投影到 embed_dim
        B, C_cnn, Hp, Wp = e_c.shape
        e_cnn = e_c.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C_cnn)  # (B, T_cnn, C_cnn)
        e_cnn = self.cnn_proj(e_cnn)                         # (B, T_cnn, D)

        # ── Step B：ViT 路径 ──
        x = self.token_encoder(
            token_raw, is_masked, pe_spatial, pe_spectral,
        )                                                    # (B, T, D)
        z_seq = self.vit(x)                                  # (B, T, D)

        # ── Step C：CIAM 双向交叉注意力 ──
        z_out, e_out = self.ciam(z_seq, e_cnn)               # (B, T, D), (B, T_cnn, D)

        # ── Step D：reshape → 预测头 ──
        z_grid = seq_to_grid(z_out, h_p, w_p, n_sp)         # (B, H_p, W_p, n_sp, D)

        # DINO-style Token 一致性投影头 [Loss 4]
        proj_s, proj_t = self.token_cons_head(z_grid, batch["c_star_patch"])

        # 跨谱聚合 + PixelDecoder（利用 CNN skip）[Loss 3]
        f_low = self.spectral_aggregate(z_grid)              # (B, H_p, W_p, D)
        # e_out 为 CNN token 经 ViT 全局语义增强后的结果 (B, T_cnn, D)；
        # 将其 reshape 为 patch 网格并与 f_low 相加，使方向 B 的参数参与梯度流。
        e_out_grid = e_out.view(B, h_p, w_p, self.embed_dim)  # (B, H_p, W_p, D)
        f_low = f_low + e_out_grid                            # 残差融合
        c_pix_raw = self.pixel_decoder(f_low, skips)         # (B, K, H, W)
        c_pix = self._activate_abundance(c_pix_raw)          # softmax / softplus

        # 物理解码 [Loss 1, 2]
        od_hat = reconstruct_od(c_pix, e_star)
        i_hat = reconstruct_intensity(od_hat, od_max=self.od_max)

        return {
            "x_embed": x,
            "z_seq": z_seq,
            "z_out": z_out,
            "z_grid": z_grid,
            "proj_s": proj_s,
            "proj_t": proj_t,
            "f_low": f_low,
            "c_pix": c_pix,
            "od_hat": od_hat,
            "i_hat": i_hat,
            "H_p": torch.tensor(h_p),
            "W_p": torch.tensor(w_p),
        }


if __name__ == "__main__":
    import math

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    B, S, H, W = 2, 60, 256, 256
    P, N_SP, S_P, K, D = 16, 6, 10, 16, 256

    model = NMFPretrainModelCINET(
        embed_dim=D, spectral_patch_size=S_P, num_endmembers=K,
        num_spectral_groups=N_SP, patch_size=P,
        vit_depth=4, vit_heads=8,
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nNMFPretrainModelCINET — 总参数量: {total / 1e6:.2f} M")

    h_p, w_p = H // P, W // P
    T = h_p * w_p * N_SP
    batch = {
        "token_raw":    torch.randn(B, T, P * P * S_P),
        "is_masked":    torch.rand(B, T) < 0.4,
        "pe_spatial":   torch.rand(B, T, 2),
        "pe_spectral":  torch.rand(B, T),
        "od":           torch.rand(B, S, H, W),
        "e_star":       torch.rand(K, S),
        "c_star_patch": torch.softmax(
            torch.randn(B, H // P, W // P, P * P * K), dim=-1
        ),
        "H": H, "W": W,
    }
    with torch.no_grad():
        out = model(batch)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<12} {tuple(v.shape)}")
