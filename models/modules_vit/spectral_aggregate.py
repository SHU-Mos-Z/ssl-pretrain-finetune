"""ViT 输出在谱段维 n_sp 上的跨视角聚合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAggregate(nn.Module):
    """
    (B, H_p, W_p, n_sp, D) → (B, H_p, W_p, D)
    """

    def __init__(self, embed_dim: int, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError(f"mode must be mean or attention, got {mode}")
        self.mode = mode
        if mode == "attention":
            self.attn_fc = nn.Linear(embed_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return z.mean(dim=3)

        # attention over n_sp (dim=3)
        scores = self.attn_fc(z).squeeze(-1)           # (B, H_p, W_p, n_sp)
        weights = F.softmax(scores, dim=3).unsqueeze(-1)
        return (z * weights).sum(dim=3)
