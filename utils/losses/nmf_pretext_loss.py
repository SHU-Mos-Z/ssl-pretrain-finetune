"""
NMF 前置 ViT 预训练损失。

L = λ_od·L_od + λ_i·L_i + λ_pix·L_cons_pix
    [+ λ_tok·L_cons_token]   （可通过 use_cons_token=False 关闭）
    [+ λ_anch·L_anchor]      （teacher_proj 外部 NMF 丰度锚定，防止投影塌陷）

L_cons_token（DINO-style 投影空间 MSE）：
    proj_s (B, H_p, W_p, n_sp, D_L) ← Student（各谱段 token 经 student_proj）
    proj_t (B, H_p, W_p, D_L)       ← Teacher（NMF 全像素丰度经 teacher_proj）
    l_cons_token = MSE(proj_s, proj_t.unsqueeze(3).expand_as(proj_s))

L_anchor（teacher_proj 丰度解码锚定，方案 B）：
    anchor_dec: Linear(D_L → K)  —— 在损失模块中维护，将 proj_t 解码回丰度空间
    c_hat = anchor_dec(proj_t)                  # (B, H_p, W_p, K)
    c_mean = mean_over_P*P(c_star_patch)        # (B, H_p, W_p, K)
    l_anchor = MSE(c_hat, c_mean.detach())
    此项使 teacher_proj 输出必须携带足够丰度信息，防止塌陷到常数。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NMFPretextLoss(nn.Module):
    def __init__(
        self,
        lambda_od: float = 1.0,
        lambda_i: float = 1.0,
        lambda_cons_pix: float = 0.5,
        lambda_cons_token: float = 0.5,
        lambda_anchor: float = 0.1,
        use_cons_token: bool = True,
        num_endmembers: int = 2,
        proj_dim: int = 128,
    ):
        """
        Args:
            lambda_od:         OD 重建损失权重
            lambda_i:          强度重建损失权重
            lambda_cons_pix:   像素级一致性损失权重
            lambda_cons_token: Token 级 DINO 一致性损失权重
            lambda_anchor:     teacher_proj 锚定损失权重（use_cons_token=True 且 >0 时生效）
            use_cons_token:    是否启用 l_cons_token + l_anchor（False 时完全跳过）
            num_endmembers:    端元数 K
            proj_dim:          TokenConsistencyHead 的投影维度 D_L（须与模型一致，默认 128）
        """
        super().__init__()
        self.lambda_od = lambda_od
        self.lambda_i = lambda_i
        self.lambda_cons_pix = lambda_cons_pix
        self.lambda_cons_token = lambda_cons_token
        self.lambda_anchor = lambda_anchor
        self.use_cons_token = use_cons_token
        self.num_endmembers = num_endmembers

        # 锚定解码头：将 teacher_proj 输出(D_L)解码回丰度空间(K)
        # 仅在 use_cons_token=True 且 lambda_anchor>0 时有意义
        self.anchor_dec: nn.Linear | None = (
            nn.Linear(proj_dim, num_endmembers)
            if use_cons_token and lambda_anchor > 0.0
            else None
        )

    def forward(
        self,
        model_out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        od_hat = model_out["od_hat"]
        i_hat = model_out["i_hat"]
        c_pix = model_out["c_pix"]

        od_gt = batch["od"]
        i_gt = batch["intensity"]
        m_pix = batch["m_pix"]          # (B, S, H, W), 1=可见, 0=loss
        c_star = batch["c_star"].detach()

        inv_m = (1.0 - m_pix).clamp(min=0.0)
        n_recon = inv_m.sum().clamp(min=1.0)

        l_od = (inv_m * (od_hat - od_gt).pow(2)).sum() / n_recon
        l_i = (inv_m * (i_hat - i_gt).pow(2)).sum() / n_recon
        l_cons_pix = (c_pix - c_star).pow(2).mean()

        total = self.lambda_od * l_od + self.lambda_i * l_i + self.lambda_cons_pix * l_cons_pix

        l_cons_token_val = 0.0
        l_anchor_val = 0.0

        if self.use_cons_token:
            # ── DINO-style Token 一致性损失 ──────────────────────────────────
            proj_s = model_out["proj_s"]   # (B, H_p, W_p, n_sp, D_L)
            proj_t = model_out["proj_t"]   # (B, H_p, W_p, D_L)

            target_tok = proj_t.unsqueeze(3).expand_as(proj_s)
            l_cons_token = (proj_s - target_tok).pow(2).mean()
            total = total + self.lambda_cons_token * l_cons_token
            l_cons_token_val = l_cons_token.item()

            # ── teacher_proj 锚定损失（方案 B）───────────────────────────────
            # anchor_dec 把 proj_t (D_L) 解码回丰度空间 (K)，
            # 再与 patch 内 P*P 像素均值丰度 c_mean 做 MSE。
            # 使 teacher_proj 必须保留丰度信息，防止塌陷。
            if self.lambda_anchor > 0.0 and self.anchor_dec is not None:
                c_star_patch = batch["c_star_patch"].detach()   # (B, H_p, W_p, P*P*K)
                B, Hp, Wp, PPK = c_star_patch.shape
                K = self.num_endmembers
                # (B, H_p, W_p, P*P, K) → mean → (B, H_p, W_p, K)
                c_mean = c_star_patch.view(B, Hp, Wp, PPK // K, K).mean(dim=3)
                # anchor_dec(proj_t): (B, H_p, W_p, D_L) → (B, H_p, W_p, K)
                c_hat = self.anchor_dec(proj_t)
                l_anchor = (c_hat - c_mean).pow(2).mean()
                total = total + self.lambda_anchor * l_anchor
                l_anchor_val = l_anchor.item()

        logs = {
            "loss_total": total.item(),
            "loss_od": l_od.item(),
            "loss_i": l_i.item(),
            "loss_cons_pix": l_cons_pix.item(),
            "loss_cons_token": l_cons_token_val,
            "loss_anchor": l_anchor_val,
        }
        return total, logs


def _self_test() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.nmf_pretrain_model_vit import NMFPretrainModel, _make_dummy_batch

    print(f"\n{'=' * 72}")
    print("  NMFPretextLoss 自检（forward + backward）")
    print(f"{'=' * 72}")

    model = NMFPretrainModel(vit_depth=2, num_spectral_groups=6, spectral_patch_size=10)
    criterion = NMFPretextLoss(
        use_cons_token=True, lambda_anchor=0.1, num_endmembers=2,
    )
    # _make_dummy_batch 已包含正确形状的 c_star_patch (B, H_p, W_p, P*P*K)
    batch = _make_dummy_batch(b=2, s=60, h=128, w=128, p=16, n_sp=6, s_p=10)
    # m_pix 须为 (B, S, H, W)
    m = batch["m_pix"]
    if m.shape[1] == 1:
        batch["m_pix"] = m.expand(-1, 60, -1, -1).clone()
        batch["m_pix"][:, :, ::4, ::4] = 0.0

    model.train()
    out = model(batch)
    loss, logs = criterion(out, batch)
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  loss_total         {logs['loss_total']:.6f}")
    for k, v in logs.items():
        if k != "loss_total":
            print(f"  {k:<18} {v:.6f}")
    print(f"  params w/ grad     {n_grad}")
    assert loss.item() > 0
    assert n_grad > 0
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    _self_test()
