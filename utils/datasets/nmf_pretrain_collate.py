"""NMF 预训练 collate：stack 固定尺寸 batch。"""

from __future__ import annotations

from typing import Any

import torch


def nmf_pretrain_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    将 NMFPretrainDataset 样本 list 合并为 model / loss 所需 batch dict。

    要求同 batch 内 H,W,S,T 一致（固定分辨率与 n_sp 时自然满足）。
    """
    h = batch[0]["H"]
    w = batch[0]["W"]
    for s in batch[1:]:
        if s["H"] != h or s["W"] != w:
            raise ValueError("batch 内 H/W 须一致，请使用相同空间分辨率数据")

    out: dict[str, Any] = {
        "H": h,
        "W": w,
        "stem": [s["stem"] for s in batch],
    }
    tensor_keys = [
        "token_raw", "is_masked", "pe_spatial", "pe_spectral", "pe_abund",
        "od", "intensity", "c_star", "c_star_patch", "e_star", "m_pix",
    ]
    for k in tensor_keys:
        out[k] = torch.stack([s[k] for s in batch], dim=0)
    return out
