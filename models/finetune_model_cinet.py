"""
CINET 版下游分割微调模型。

前向流程（无 masking）：
  1. ContextualEncoder(CNN) → e_c + skips
  2. TokenEncoder(3D tokenize, 无掩膜) → ViT → z_seq
  3. CIAM(z_vit, e_cnn) → z_out
  4. reshape → z_grid (B, H_p, W_p, n_sp, D)
  5. SpectralAggregate → f_low (B, H_p, W_p, D)
  6. PixelDecoder(f_low, skips) → logits (B, num_classes, H, W)

backbone 权重（cnn_encoder / token_encoder / vit / ciam / spectral_aggregate）
可从 NMFPretrainModelCINET checkpoint 加载。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.modules_vit.spectral_aggregate import SpectralAggregate
from models.modules_vit.token_encoder import TokenEncoder
from models.modules_vit.vit_backbone import ViTBackbone
from models.modules_cinet.ciam import CIAM
from models.modules_cinet.contextual_encoder import ContextualEncoder
from models.modules_cinet.pixel_decoder import PixelDecoder
from models.nmf_pretrain_model_vit import seq_to_grid


class FinetuneModelCINET(nn.Module):
    """
    CINET 微调模型：backbone 与预训练版共享架构，最终输出分割 logits。

    Args:
        num_classes     : 分割类别数。
        embed_dim       : ViT 隐藏维度 D。
        spectral_patch_size : 每谱段组的波段数 s_p。
        num_endmembers  : NMF 端元数 K（仅用于 TokenEncoder 内部 abund_proj，微调时通常不启用）。
        num_spectral_groups : 谱段组数 n_sp。
        patch_size      : 空间 Patch 边长 P（须为 2 的整数次幂）。
        vit_depth       : ViT Transformer 层数。
        vit_heads       : ViT 多头注意力头数。
        mlp_ratio       : ViT FFN 隐藏层倍率。
        dropout         : ViT dropout。
        use_abund_pe    : 微调阶段通常置 False（不依赖 NMF 丰度位置编码）。
        aggregate_mode  : 跨谱聚合方式（'mean' 或 'attention'）。
        cnn_stem_ch     : CNN Stem 输出通道数。
        cnn_spectral_agg: CNN SpectralAggregator 聚合方式。
        ciam_heads      : CIAM 多头注意力头数。
        ciam_dropout    : CIAM attention dropout。
        ciam_ffn_ratio  : CIAM FFN 隐藏层倍率。
        decoder_mid_ch  : PixelDecoder final_conv 中间通道数。
        pretrain_ckpt   : 预训练 checkpoint 路径（可选）。
        freeze_backbone : 是否冻结 backbone（Linear Probe 设为 True）。
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        spectral_patch_size: int = 10,
        num_endmembers: int = 2,
        num_spectral_groups: int = 6,
        patch_size: int = 16,
        vit_depth: int = 6,
        vit_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_abund_pe: bool = False,
        aggregate_mode: str = "mean",
        cnn_stem_ch: int = 64,
        cnn_spectral_agg: str = "attention",
        ciam_heads: int = 8,
        ciam_dropout: float = 0.1,
        ciam_ffn_ratio: float = 2.0,
        decoder_mid_ch: int = 64,
        pretrain_ckpt: str | None = None,
        freeze_backbone: bool = False,
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
        self.num_classes = num_classes

        # ── ViT 路径 ──
        self.token_encoder = TokenEncoder(
            embed_dim=embed_dim,
            spectral_patch_size=spectral_patch_size,
            num_endmembers=num_endmembers,
            use_abund_pe=use_abund_pe,
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
        self.cnn_proj = nn.Linear(self.cnn_encoder.out_ch, embed_dim)

        # ── 跨谱聚合 ──
        self.spectral_aggregate = SpectralAggregate(embed_dim, mode=aggregate_mode)

        # ── 分割 Decoder（输出 num_classes 通道） ──
        self.pixel_decoder = PixelDecoder(
            vit_dim=embed_dim,
            stem_ch=cnn_stem_ch,
            layer_channels=self.cnn_encoder.layer_channels,
            out_ch=num_classes,
            mid_ch=decoder_mid_ch,
        )

        if pretrain_ckpt is not None:
            self.load_pretrain(pretrain_ckpt)
        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def grid_size(h: int, w: int, patch_size: int) -> tuple[int, int]:
        assert h % patch_size == 0 and w % patch_size == 0
        return h // patch_size, w // patch_size

    def load_pretrain(self, ckpt_path: str) -> None:
        """
        从 NMFPretrainModelCINET checkpoint 加载 backbone 权重。
        加载的模块：token_encoder / vit / cnn_encoder / cnn_proj / ciam / spectral_aggregate。
        不加载 pixel_decoder（输出通道不同：预训练为 K，微调为 num_classes）。
        """
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        backbone_prefixes = (
            "token_encoder.", "vit.",
            "cnn_encoder.", "cnn_proj.",
            "ciam.", "spectral_aggregate.",
        )
        filtered = {k: v for k, v in state.items() if k.startswith(backbone_prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"[FinetuneModelCINET] 加载预训练: {ckpt_path}\n"
            f"  匹配加载: {len(filtered)}  缺失: {len(missing)}  意外: {len(unexpected)}"
        )

    def freeze_backbone(self) -> None:
        """冻结除 pixel_decoder 外的所有模块（Linear Probe 模式）。"""
        for module in [
            self.token_encoder, self.vit,
            self.cnn_encoder, self.cnn_proj,
            self.ciam, self.spectral_aggregate,
        ]:
            for p in module.parameters():
                p.requires_grad_(False)
        print("[FinetuneModelCINET] backbone 已冻结（Linear Probe）")

    def forward_features(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        batch 必含字段：
            token_raw   (B, T, s_p)
            is_masked   (B, T) bool  — 微调时全为 False
            pe_spatial  (B, T, 2)
            pe_spectral (B, T)
            od          (B, S, H, W)  — 微调时无需遮蔽，直接送 CNN
            H, W        int
        """
        h, w = int(batch["H"]), int(batch["W"])
        h_p, w_p = self.grid_size(h, w, self.patch_size)
        n_sp = self.num_spectral_groups
        od = batch["od"]

        # CNN 路径：微调时不遮蔽，直接使用完整 OD
        e_c, skips = self.cnn_encoder(od)
        B, C_cnn, Hp, Wp = e_c.shape
        e_cnn = e_c.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C_cnn)
        e_cnn = self.cnn_proj(e_cnn)                          # (B, T_cnn, D)

        # ViT 路径
        x = self.token_encoder(
            batch["token_raw"], batch["is_masked"],
            batch["pe_spatial"], batch["pe_spectral"],
            pe_abund=batch.get("pe_abund"), w_abund=w_abund,
        )
        z_seq = self.vit(x)

        # CIAM
        z_out, e_out = self.ciam(z_seq, e_cnn)

        # reshape
        z_grid = seq_to_grid(z_out, h_p, w_p, n_sp)           # (B, H_p, W_p, n_sp, D)
        f_low = self.spectral_aggregate(z_grid)                # (B, H_p, W_p, D)
        # e_out 为 CNN token 经 ViT 全局语义增强后的结果；融合进 f_low，
        # 使方向 B 的 CIAM 参数参与梯度，同时为 PixelDecoder 提供额外语义。
        e_out_grid = e_out.view(B, h_p, w_p, self.embed_dim)  # (B, H_p, W_p, D)
        f_low = f_low + e_out_grid

        return {
            "x_embed": x,
            "z_seq": z_seq,
            "z_out": z_out,
            "z_grid": z_grid,
            "f_low": f_low,
            "skips": skips,
        }

    def forward(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> torch.Tensor:
        feats = self.forward_features(batch, w_abund=w_abund)
        logits = self.pixel_decoder(feats["f_low"], feats["skips"])
        return logits

    def forward_with_features(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, w_abund=w_abund)
        logits = self.pixel_decoder(feats["f_low"], feats["skips"])
        return logits, feats


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    B, S, H, W = 2, 60, 256, 256
    P, N_SP, S_P, K, D = 16, 6, 10, 2, 256
    NC = 3

    model = FinetuneModelCINET(
        num_classes=NC, embed_dim=D, spectral_patch_size=S_P,
        num_endmembers=K, num_spectral_groups=N_SP, patch_size=P,
        vit_depth=4, vit_heads=8, use_abund_pe=False,
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nFinetuneModelCINET — 总参数量: {total / 1e6:.2f} M")

    h_p, w_p = H // P, W // P
    T = h_p * w_p * N_SP
    batch = {
        "token_raw":   torch.randn(B, T, S_P),
        "is_masked":   torch.zeros(B, T, dtype=torch.bool),
        "pe_spatial":  torch.rand(B, T, 2),
        "pe_spectral": torch.rand(B, T),
        "od":          torch.rand(B, S, H, W),
        "H": H, "W": W,
    }
    with torch.no_grad():
        logits, feats = model.forward_with_features(batch)
    print(f"  logits  {tuple(logits.shape)}")
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<12} {tuple(v.shape)}")
