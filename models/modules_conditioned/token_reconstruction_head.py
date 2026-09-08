import torch
import torch.nn as nn


class TokenReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, token_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(), nn.Linear(2 * embed_dim, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)
