"""
下游微调分割模型：复用预训练 TokenEncoder + ViT + SpectralAggregate，
上采样得 f_full (B,D,H,W) 接轻量分割头。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any

import torch
import torch.nn as nn

from models.modules_vit.abundance_head import upsample_features
from models.modules_vit.spectral_aggregate import SpectralAggregate
from models.modules_vit.token_encoder import TokenEncoder
from models.modules_vit.vit_backbone import ViTBackbone
from models.nmf_pretrain_model_vit import seq_to_grid


class SegmentationHead(nn.Module):
    """Conv-BN-ReLU-Conv 分割头，输入 (B, D, H, W) → (B, num_classes, H, W)。"""

    def __init__(self, in_channels: int, num_classes: int, mid_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FinetuneModel(nn.Module):
    """
    微调路径 A：与预训练共享 TokenEncoder / ViT / SpectralAggregate，
    聚合特征上采样至全图后接分割头。
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
        pretrain_ckpt: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spectral_patch_size = spectral_patch_size
        self.num_endmembers = num_endmembers
        self.num_spectral_groups = num_spectral_groups
        self.patch_size = patch_size
        self.num_classes = num_classes

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
        self.spectral_aggregate = SpectralAggregate(embed_dim, mode=aggregate_mode)
        self.seg_head = SegmentationHead(embed_dim, num_classes)

        if pretrain_ckpt is not None:
            self.load_pretrain(pretrain_ckpt)
        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def grid_size(h: int, w: int, patch_size: int) -> tuple[int, int]:
        assert h % patch_size == 0 and w % patch_size == 0
        return h // patch_size, w // patch_size

    def load_pretrain(self, ckpt_path: str) -> None:
        """加载 NMFPretrainModel 的 backbone 权重（token_encoder / vit / spectral_aggregate）。"""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        prefixes = ("token_encoder.", "vit.", "spectral_aggregate.")
        filtered = {k: v for k, v in state.items() if k.startswith(prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"[FinetuneModel] 加载预训练: {ckpt_path}\n"
            f"  匹配加载: {len(filtered)}  缺失: {len(missing)}  意外: {len(unexpected)}"
        )

    def freeze_backbone(self) -> None:
        for module in [self.token_encoder, self.vit, self.spectral_aggregate]:
            for p in module.parameters():
                p.requires_grad_(False)
        print("[FinetuneModel] backbone 已冻结（Linear Probe）")

    def forward_features(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        h, w = int(batch["H"]), int(batch["W"])
        h_p, w_p = self.grid_size(h, w, self.patch_size)
        n_sp = self.num_spectral_groups

        x = self.token_encoder(
            batch["token_raw"], batch["is_masked"],
            batch["pe_spatial"], batch["pe_spectral"],
            pe_abund=batch.get("pe_abund"), w_abund=w_abund,
        )
        z_seq = self.vit(x)
        z_grid = seq_to_grid(z_seq, h_p, w_p, n_sp)
        f_low = self.spectral_aggregate(z_grid)
        f_full = upsample_features(f_low, (h, w))
        return {
            "x_embed": x,
            "z_seq": z_seq,
            "z_grid": z_grid,
            "f_low": f_low,
            "f_full": f_full,
        }

    def forward(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> torch.Tensor:
        feats = self.forward_features(batch, w_abund=w_abund)
        return self.seg_head(feats["f_full"])

    def forward_with_features(
        self,
        batch: dict[str, Any],
        w_abund: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, w_abund=w_abund)
        logits = self.seg_head(feats["f_full"])
        return logits, feats


def _fmt(t: torch.Tensor | tuple) -> str:
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    return str(tuple(t))


def _make_dummy_batch(
    b: int = 2,
    h: int = 256,
    w: int = 256,
    p: int = 16,
    n_sp: int = 6,
    k: int = 2,
    s_p: int = 10,
    mask_ratio: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    h_p, w_p = h // p, w // p
    t = h_p * w_p * n_sp

    token_raw = torch.randn(b, t, s_p, device=device)
    is_masked = torch.rand(b, t, device=device) < mask_ratio

    u_idx = torch.arange(h_p, device=device).view(h_p, 1, 1).expand(h_p, w_p, n_sp)
    v_idx = torch.arange(w_p, device=device).view(1, w_p, 1).expand(h_p, w_p, n_sp)
    j_idx = torch.arange(n_sp, device=device).view(1, 1, n_sp).expand(h_p, w_p, n_sp)
    x_c = (u_idx.float() + 0.5) * p / w
    y_c = (v_idx.float() + 0.5) * p / h
    pe_spatial = torch.stack([x_c, y_c], dim=-1).reshape(1, t, 2).expand(b, -1, -1)
    pe_spectral = (j_idx.float() + 0.5) / n_sp
    pe_spectral = pe_spectral.reshape(1, t).expand(b, -1)

    return {
        "token_raw": token_raw,
        "is_masked": is_masked,
        "pe_spatial": pe_spatial,
        "pe_spectral": pe_spectral,
        "H": h,
        "W": w,
    }


def trace_finetune_forward(model: FinetuneModel, batch: dict[str, Any]) -> None:
    h, w = int(batch["H"]), int(batch["W"])
    h_p, w_p = model.grid_size(h, w, model.patch_size)
    n_sp = model.num_spectral_groups
    t = h_p * w_p * n_sp
    nc = model.num_classes
    d = model.embed_dim

    print(f"\n{'─' * 78}")
    print("  FinetuneModel 前向传播 — 关键张量尺寸")
    print(f"  配置: H={h}, W={w}, P={model.patch_size}, H_p={h_p}, W_p={w_p}, "
          f"n_sp={n_sp}, T={t}, D={d}, num_classes={nc}")
    print(f"{'─' * 78}")

    print(f"  [batch] token_raw      {_fmt(batch['token_raw'])}")
    print(f"  [batch] is_masked      {_fmt(batch['is_masked'])}")
    print(f"  [batch] pe_spatial     {_fmt(batch['pe_spatial'])}")
    print(f"  [batch] pe_spectral    {_fmt(batch['pe_spectral'])}")

    with torch.no_grad():
        x = model.token_encoder(
            batch["token_raw"], batch["is_masked"],
            batch["pe_spatial"], batch["pe_spectral"],
        )
        print(f"  [TokenEncoder]         →  x_embed={_fmt(x)}")

        z_seq = model.vit(x)
        print(f"  [ViTBackbone]          →  z_seq={_fmt(z_seq)}")

        z_grid = seq_to_grid(z_seq, h_p, w_p, n_sp)
        print(f"  [reshape]              →  z_grid={_fmt(z_grid)}")

        f_low = model.spectral_aggregate(z_grid)
        print(f"  [SpectralAggregate]    →  f_low={_fmt(f_low)}")

        f_full = upsample_features(f_low, (h, w))
        print(f"  [upsample_features]    →  f_full={_fmt(f_full)}")

        logits = model.seg_head(f_full)
        print(f"  [SegmentationHead]     →  logits={_fmt(logits)}")

        logits2, feats = model.forward_with_features(batch)
        print(f"\n  [model.forward 对照]   logits={_fmt(logits2)}")
        for key in ["x_embed", "z_seq", "z_grid", "f_low", "f_full"]:
            print(f"    {key:<12} {_fmt(feats[key])}")

        assert logits.shape == logits2.shape == (batch["token_raw"].shape[0], nc, h, w)

    print(f"{'─' * 78}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    B, H, W = 2, 256, 256
    P, N_SP, S_P, K, D = 16, 6, 10, 2, 256
    NUM_CLASSES = 2

    model = FinetuneModel(
        num_classes=NUM_CLASSES,
        embed_dim=D,
        spectral_patch_size=S_P,
        num_endmembers=K,
        num_spectral_groups=N_SP,
        patch_size=P,
        vit_depth=4,
        vit_heads=8,
        use_abund_pe=False,
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 78}")
    print("  FinetuneModel — 前向 shape 诊断")
    print(f"  B={B}, H×W={H}×{W}, num_classes={NUM_CLASSES}, D={D}")
    print(f"  总参数量: {total / 1e6:.2f} M")
    print(f"{'=' * 78}")

    batch = _make_dummy_batch(b=B, h=H, w=W, p=P, n_sp=N_SP, s_p=S_P)
    trace_finetune_forward(model, batch)

    print("\n【附加】H=W=128, B=1")
    batch_small = _make_dummy_batch(b=1, h=128, w=128, p=P, n_sp=N_SP, s_p=S_P)
    trace_finetune_forward(model, batch_small)

    print(f"\n{'=' * 78}")
    print("  微调模型前向诊断完成。")
    print(f"{'=' * 78}\n")
