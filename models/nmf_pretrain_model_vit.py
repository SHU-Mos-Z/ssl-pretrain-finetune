"""
NMF 前置 + 规则 Patch × 谱段组 ViT 预训练模型。

前向流程：TokenEncoder → ViT → reshape → TokenConsHead / SpectralAggregate
→ AbundanceHead → physics decode（OD / I 重建）。
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

from models.modules_vit.abundance_head import AbundanceHead
from models.modules_vit.physics_decode import reconstruct_intensity, reconstruct_od
from models.modules_vit.spectral_aggregate import SpectralAggregate
from models.modules_vit.token_consistency_head import TokenConsistencyHead
from models.modules_vit.token_encoder import TokenEncoder
from models.modules_vit.vit_backbone import ViTBackbone


def seq_to_grid(z: torch.Tensor, h_p: int, w_p: int, n_sp: int) -> torch.Tensor:
    """
    (B, T, D) → (B, H_p, W_p, n_sp, D)，flatten 顺序 t = u*(W_p*n_sp) + v*n_sp + j。
    """
    b, t, d = z.shape
    assert t == h_p * w_p * n_sp, f"T={t} != H_p*W_p*n_sp={h_p * w_p * n_sp}"
    return z.view(b, h_p, w_p, n_sp, d)


class NMFPretrainModel(nn.Module):
    def __init__(
        self,
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
        use_refine: bool = True,
        od_max: float = 3.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spectral_patch_size = spectral_patch_size
        self.num_endmembers = num_endmembers
        self.num_spectral_groups = num_spectral_groups
        self.patch_size = patch_size
        self.od_max = od_max

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
        self.token_cons_head = TokenConsistencyHead(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_endmembers=num_endmembers,
        )
        self.spectral_aggregate = SpectralAggregate(embed_dim, mode=aggregate_mode)
        self.abundance_head = AbundanceHead(
            embed_dim=embed_dim,
            num_endmembers=num_endmembers,
            use_refine=use_refine,
            activation=abundance_activation,
        )

    @staticmethod
    def grid_size(h: int, w: int, patch_size: int) -> tuple[int, int]:
        assert h % patch_size == 0 and w % patch_size == 0
        return h // patch_size, w // patch_size

    def forward(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch 必含字段：
                token_raw     (B, T, P*P*s_p)
                is_masked     (B, T) bool
                pe_spatial    (B, T, 2)
                pe_spectral   (B, T)
                e_star        (K, S)
                c_star_patch  (B, H_p, W_p, P*P*K)
                H, W          int
        """
        token_raw = batch["token_raw"]
        is_masked = batch["is_masked"]
        pe_spatial = batch["pe_spatial"]
        pe_spectral = batch["pe_spectral"]
        e_star = batch["e_star"]
        h, w = int(batch["H"]), int(batch["W"])

        h_p, w_p = self.grid_size(h, w, self.patch_size)
        n_sp = self.num_spectral_groups

        # Step 4: Token 嵌入
        x = self.token_encoder(token_raw, is_masked, pe_spatial, pe_spectral)

        # Step 5: ViT + reshape
        z_seq = self.vit(x)
        z_grid = seq_to_grid(z_seq, h_p, w_p, n_sp)

        # Step 6.1: DINO-style Token 一致性投影头
        c_star_patch = batch["c_star_patch"]
        proj_s, proj_t = self.token_cons_head(z_grid, c_star_patch)

        # Step 6.2: 跨谱聚合 + 丰度头
        f_low = self.spectral_aggregate(z_grid)
        c_pix = self.abundance_head(f_low, (h, w))

        # Step 6.3: 物理解码
        od_hat = reconstruct_od(c_pix, e_star)
        i_hat = reconstruct_intensity(od_hat, od_max=self.od_max)

        return {
            "x_embed": x,
            "z_seq": z_seq,
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


def _fmt(t: torch.Tensor | tuple) -> str:
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    return str(tuple(t))


def _make_dummy_batch(
    b: int = 2,
    s: int = 60,
    h: int = 256,
    w: int = 256,
    p: int = 16,
    n_sp: int = 6,
    k: int = 2,
    s_p: int = 10,
    mask_ratio: float = 0.4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """合成与训练 collate 一致的 batch，供 shape 诊断。"""
    device = device or torch.device("cpu")
    h_p, w_p = h // p, w // p
    t = h_p * w_p * n_sp

    token_raw = torch.randn(b, t, p * p * s_p, device=device)
    is_masked = torch.rand(b, t, device=device) < mask_ratio

    u_idx = torch.arange(h_p, device=device).view(h_p, 1, 1).expand(h_p, w_p, n_sp)
    v_idx = torch.arange(w_p, device=device).view(1, w_p, 1).expand(h_p, w_p, n_sp)
    j_idx = torch.arange(n_sp, device=device).view(1, 1, n_sp).expand(h_p, w_p, n_sp)

    x_c = (u_idx.float() + 0.5) * p / w
    y_c = (v_idx.float() + 0.5) * p / h
    pe_spatial = torch.stack([x_c, y_c], dim=-1).reshape(1, t, 2).expand(b, -1, -1)
    pe_spectral = (j_idx.float() + 0.5) / n_sp
    pe_spectral = pe_spectral.reshape(1, t).expand(b, -1)
    e_star = torch.randn(k, s, device=device).abs()
    e_star = e_star / e_star.norm(dim=1, keepdim=True).clamp(min=1e-6)

    return {
        "token_raw": token_raw,
        "is_masked": is_masked,
        "pe_spatial": pe_spatial,
        "pe_spectral": pe_spectral,
        "e_star": e_star,
        "H": h,
        "W": w,
        "od": torch.randn(b, s, h, w, device=device).abs(),
        "intensity": torch.rand(b, s, h, w, device=device),
        "c_star": torch.softmax(torch.randn(b, k, h, w, device=device), dim=1),
        "c_star_patch": torch.softmax(
            torch.randn(b, h // p, w // p, p * p * k, device=device), dim=-1
        ),
        "m_pix": (torch.rand(b, 1, h, w, device=device) > 0.5).float(),
    }


def trace_pretrain_forward(model: NMFPretrainModel, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """逐步前向并打印所有关键张量尺寸。"""
    h, w = int(batch["H"]), int(batch["W"])
    h_p, w_p = model.grid_size(h, w, model.patch_size)
    n_sp = model.num_spectral_groups
    k = model.num_endmembers
    s_p = model.spectral_patch_size
    s = batch["e_star"].shape[1]
    t = h_p * w_p * n_sp

    print(f"\n{'─' * 78}")
    print("  NMFPretrainModel 前向传播 — 关键张量尺寸")
    print(f"  配置: H={h}, W={w}, P={model.patch_size}, H_p={h_p}, W_p={w_p}, "
          f"n_sp={n_sp}, s_p={s_p}, K={k}, S={s}, T={t}, D={model.embed_dim}")
    print(f"{'─' * 78}")

    b = batch["token_raw"].shape[0]
    print(f"  [batch] token_raw      {_fmt(batch['token_raw'])}")
    print(f"  [batch] is_masked      {_fmt(batch['is_masked'])}  (masked={batch['is_masked'].sum().item()}/{batch['is_masked'].numel()})")
    print(f"  [batch] pe_spatial     {_fmt(batch['pe_spatial'])}")
    print(f"  [batch] pe_spectral    {_fmt(batch['pe_spectral'])}")
    print(f"  [batch] e_star         {_fmt(batch['e_star'])}")
    if "od" in batch:
        print(f"  [batch] od (GT)        {_fmt(batch['od'])}")
    if "c_star" in batch:
        print(f"  [batch] c_star (GT)    {_fmt(batch['c_star'])}")
    if "m_pix" in batch:
        print(f"  [batch] m_pix          {_fmt(batch['m_pix'])}")

    with torch.no_grad():
        x = model.token_encoder(
            batch["token_raw"], batch["is_masked"],
            batch["pe_spatial"], batch["pe_spectral"],
        )
        print(f"  [TokenEncoder]         token_raw={_fmt(batch['token_raw'])}  →  x_embed={_fmt(x)}")

        z_seq = model.vit(x)
        print(f"  [ViTBackbone]          x_embed={_fmt(x)}  →  z_seq={_fmt(z_seq)}")

        z_grid = seq_to_grid(z_seq, h_p, w_p, n_sp)
        print(f"  [reshape]              z_seq={_fmt(z_seq)}  →  z_grid={_fmt(z_grid)}")

        proj_s, proj_t = model.token_cons_head(z_grid, batch["c_star_patch"])
        print(f"  [TokenConsHead]        z_grid={_fmt(z_grid)}  →  proj_s={_fmt(proj_s)}, proj_t={_fmt(proj_t)}")

        f_low = model.spectral_aggregate(z_grid)
        print(f"  [SpectralAggregate]    z_grid={_fmt(z_grid)}  →  f_low={_fmt(f_low)}")

        c_pix = model.abundance_head(f_low, (h, w))
        print(f"  [AbundanceHead]        f_low={_fmt(f_low)}  →  c_pix={_fmt(c_pix)}")

        od_hat = reconstruct_od(c_pix, batch["e_star"])
        print(f"  [reconstruct_od]       c_pix={_fmt(c_pix)}, e*={_fmt(batch['e_star'])}  →  od_hat={_fmt(od_hat)}")

        i_hat = reconstruct_intensity(od_hat, od_max=model.od_max)
        print(f"  [reconstruct_I]        od_hat={_fmt(od_hat)}  →  i_hat={_fmt(i_hat)}")

        out = model(batch)
        print(f"\n  [model.forward 对照]")
        for key in ["x_embed", "z_seq", "z_grid", "proj_s", "proj_t", "f_low", "c_pix", "od_hat", "i_hat"]:
            print(f"    {key:<12} {_fmt(out[key])}")

    print(f"{'─' * 78}")
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # ── 默认配置（与方法文档一致） ──
    B, S, H, W = 2, 60, 256, 256
    P, N_SP, S_P, K, D = 16, 6, 10, 2, 256

    model = NMFPretrainModel(
        embed_dim=D,
        spectral_patch_size=S_P,
        num_endmembers=K,
        num_spectral_groups=N_SP,
        patch_size=P,
        vit_depth=4,
        vit_heads=8,
        aggregate_mode="mean",
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 78}")
    print("  NMFPretrainModel — 前向 shape 诊断")
    print(f"  B={B}, S={S}, H×W={H}×{W}, P={P}, n_sp={N_SP}, K={K}, D={D}")
    print(f"  总参数量: {total / 1e6:.2f} M")
    print(f"{'=' * 78}")

    batch = _make_dummy_batch(b=B, s=S, h=H, w=W, p=P, n_sp=N_SP, k=K, s_p=S_P)
    trace_pretrain_forward(model, batch)

    # 额外：较小分辨率快速验证
    print("\n【附加】H=W=128, B=1")
    batch_small = _make_dummy_batch(b=1, s=S, h=128, w=128, p=P, n_sp=N_SP, k=K, s_p=S_P)
    trace_pretrain_forward(model, batch_small)

    print(f"\n{'=' * 78}")
    print("  预训练模型前向诊断完成。")
    print(f"{'=' * 78}\n")
