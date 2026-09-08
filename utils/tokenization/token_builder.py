"""编排 Token 构建：特征 + 位置编码 + 教师目标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.tokenization.patch_tokens import (
    compute_patch_token_raw,
    flatten_index,
    grid_sizes,
    unflatten_index,
)
from utils.tokenization.positional_targets import (
    compute_c_star_patch,
    compute_pe_abund_from_map,
    compute_pe_spectral,
    compute_pe_spatial,
)
from utils.tokenization.spectral_groups import SpectralGroups


@dataclass
class TokenBuildConfig:
    patch_size: int = 16
    spectral_patch_size: int = 10   # s_p；n_sp = S // s_p

    def num_groups_for(self, num_bands: int) -> int:
        if num_bands % self.spectral_patch_size != 0:
            raise ValueError(
                f"S={num_bands} 须整除 s_p={self.spectral_patch_size}，"
                f"请调整 spectral_patch_size 或过滤该数据集"
            )
        return num_bands // self.spectral_patch_size


@dataclass
class TokenSample:
    """单张图 Token 化结果（numpy，collate 前）。"""

    token_raw: np.ndarray       # (T, P*P*s_p)  —— ViT 式全空间-谱段展平
    pe_spatial: np.ndarray      # (T, 2)
    pe_spectral: np.ndarray     # (T,)
    pe_abund: np.ndarray        # (T, K)
    c_star_patch: np.ndarray    # (H_p, W_p, P*P*K)  —— DINO-style 全像素丰度展平
    h_p: int
    w_p: int
    n_sp: int
    s_p: int
    h: int
    w: int
    s: int

    @property
    def num_tokens(self) -> int:
        return self.token_raw.shape[0]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def build_tokens(
    od: np.ndarray,
    c_star: np.ndarray,
    cfg: TokenBuildConfig,
) -> TokenSample:
    """
    从 OD 立方与 NMF 教师丰度构建 Token 字段。

    Args:
        od: (S, H, W)
        c_star: (K, H, W)
    """
    s, h, w = od.shape
    k = c_star.shape[0]
    n_sp = cfg.num_groups_for(s)
    groups = SpectralGroups(s, n_sp)
    h_p, w_p = grid_sizes(h, w, cfg.patch_size)

    token_raw = compute_patch_token_raw(od, cfg.patch_size, groups)
    pe_spatial = compute_pe_spatial(h, w, cfg.patch_size, groups)
    pe_spectral = compute_pe_spectral(groups, h, w, cfg.patch_size)
    pe_abund = compute_pe_abund_from_map(c_star, cfg.patch_size, groups.num_groups)
    c_star_patch = compute_c_star_patch(c_star, cfg.patch_size)

    return TokenSample(
        token_raw=token_raw,
        pe_spatial=pe_spatial,
        pe_spectral=pe_spectral,
        pe_abund=pe_abund,
        c_star_patch=c_star_patch,
        h_p=h_p,
        w_p=w_p,
        n_sp=groups.num_groups,
        s_p=groups.patch_size,
        h=h,
        w=w,
        s=s,
    )


def verify_index_roundtrip(h_p: int, w_p: int, n_sp: int) -> bool:
    """验证 flatten / unflatten 可逆。"""
    t_total = h_p * w_p * n_sp
    for t in range(t_total):
        u, v, j = unflatten_index(t, w_p, n_sp)
        if flatten_index(u, v, j, w_p, n_sp) != t:
            return False
    return True


def _self_test() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from utils.preprocessing.offline_nmf import run_offline_nmf_on_cube

    print(f"\n{'=' * 72}")
    print("  token_builder 自检")
    print(f"{'=' * 72}")

    rng = np.random.default_rng(0)
    od = rng.random((60, 128, 128)).astype(np.float32) * 2.0
    nmf = run_offline_nmf_on_cube(od, max_iter=100, verbose=False)
    cfg = TokenBuildConfig(patch_size=16, spectral_patch_size=10)
    sample = build_tokens(od, nmf.c_star, cfg)

    assert verify_index_roundtrip(sample.h_p, sample.w_p, sample.n_sp)
    assert sample.token_raw.shape == (sample.num_tokens, cfg.patch_size * cfg.patch_size * sample.s_p)
    assert sample.pe_spatial.shape == (sample.num_tokens, 2)
    assert sample.pe_spectral.shape == (sample.num_tokens,)
    assert sample.pe_abund.shape == (sample.num_tokens, nmf.c_star.shape[0])
    p = cfg.patch_size
    k = nmf.c_star.shape[0]
    assert sample.c_star_patch.shape == (sample.h_p, sample.w_p, p * p * k)

    print(f"  OD shape           {od.shape}")
    print(f"  H_p, W_p, n_sp     {sample.h_p}, {sample.w_p}, {sample.n_sp}")
    print(f"  T, s_p             {sample.num_tokens}, {sample.s_p}")
    print(f"  token_raw          {sample.token_raw.shape}")
    print(f"  pe_spatial         {sample.pe_spatial.shape}")
    print(f"  pe_spectral        {sample.pe_spectral.shape}")
    print(f"  pe_abund           {sample.pe_abund.shape}")
    print(f"  c_star_patch       {sample.c_star_patch.shape}")
    print(f"  index roundtrip    OK")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    _self_test()
