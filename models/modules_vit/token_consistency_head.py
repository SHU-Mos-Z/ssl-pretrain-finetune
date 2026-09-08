"""
DINO-style Token 级跨谱一致性投影头。

将 CIAM 输出 token（Student）与 NMF 全像素丰度展平（Teacher）
分别投影到同一 D_L 维损失空间，在该空间中计算一致性损失。

Student 路径：(B, H_p, W_p, n_sp, D)   → Linear(D → proj_dim)   → (B, H_p, W_p, n_sp, proj_dim)
Teacher 路径：(B, H_p, W_p, P*P*K)     → Linear(P*P*K → proj_dim) → (B, H_p, W_p, proj_dim)
"""

import torch
import torch.nn as nn


class TokenConsistencyHead(nn.Module):
    """
    DINO-style 双投影一致性头。

    Args:
        embed_dim:      Student token 的特征维度 D
        patch_size:     空间 Patch 大小 P
        num_endmembers: 端元数 K（NMF 丰度维度）
        proj_dim:       投影后的损失空间维度 D_L
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_endmembers: int,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Student 路径：LayerNorm + Linear
        self.student_norm = nn.LayerNorm(embed_dim)
        self.student_proj = nn.Linear(embed_dim, proj_dim)

        # Teacher 路径：对 NMF 全像素丰度展平 (P*P*K) 做线性投影
        teacher_in = patch_size * patch_size * num_endmembers
        self.teacher_proj = nn.Linear(teacher_in, proj_dim)

    def forward_student(self, z_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_grid: (B, H_p, W_p, n_sp, D)
        Returns:
            proj_s: (B, H_p, W_p, n_sp, proj_dim)
        """
        B, Hp, Wp, Nsp, D = z_grid.shape
        flat = z_grid.reshape(B * Hp * Wp * Nsp, D)
        proj = self.student_proj(self.student_norm(flat))
        return proj.view(B, Hp, Wp, Nsp, self.proj_dim)

    def forward_teacher(self, c_star_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_star_patch: (B, H_p, W_p, P*P*K)
        Returns:
            proj_t: (B, H_p, W_p, proj_dim)
        """
        return self.teacher_proj(c_star_patch)

    def forward(
        self,
        z_grid: torch.Tensor,
        c_star_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_grid:       (B, H_p, W_p, n_sp, D)
            c_star_patch: (B, H_p, W_p, P*P*K)
        Returns:
            proj_s: (B, H_p, W_p, n_sp, proj_dim)   —— Student 投影
            proj_t: (B, H_p, W_p, proj_dim)          —— Teacher 投影
        """
        proj_s = self.forward_student(z_grid)
        proj_t = self.forward_teacher(c_star_patch)
        return proj_s, proj_t
