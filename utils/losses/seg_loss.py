"""微调分割损失。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        pred = logits.argmax(dim=1)
        dice_vals = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            p = (pred == c).float()
            t = (target == c).float()
            inter = (p * t).sum()
            dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
            dice_vals.append(dice)
        dice = torch.stack(dice_vals).mean() if dice_vals else torch.tensor(0.0, device=logits.device)
        loss = self.ce_weight * ce + self.dice_weight * (1.0 - dice)
        return loss, {"loss_seg": loss.item(), "loss_ce": ce.item(), "dice": dice.item()}
