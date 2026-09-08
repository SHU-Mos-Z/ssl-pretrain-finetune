"""Pretext objectives for endmember-conditioned pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedPretextLoss(nn.Module):
    def __init__(
        self,
        lambda_od: float = 1.0,
        lambda_i: float = 1.0,
        lambda_c: float = 0.2,
        lambda_token: float = 1.0,
        lambda_feature: float = 0.0,
        lambda_delta: float = 0.01,
        lambda_sam: float = 0.0,
    ):
        super().__init__()
        self.weights = {
            "od": lambda_od, "i": lambda_i, "c": lambda_c,
            "token": lambda_token, "feature": lambda_feature,
            "delta": lambda_delta, "sam": lambda_sam,
        }

    @staticmethod
    def _masked_mse(pred, target, mask):
        mask = mask.to(pred.dtype).expand_as(pred)
        return (mask * (pred - target).square()).sum() / mask.sum().clamp(min=1.0)

    def forward(self, output: dict, batch: dict, output_b: dict | None = None):
        valid = batch["valid_voxel"].bool()
        hidden = (~batch["voxel_visible"].bool()) & valid
        l_od = self._masked_mse(output["od_hat"], batch["od"], hidden)
        l_i = self._masked_mse(output["i_hat"], batch["intensity"], hidden)

        abundance_weight = batch["w_nmf"].to(output["c_hat"].dtype) * valid
        l_c = (
            abundance_weight * (output["c_hat"] - batch["c_star"].detach()).square()
        ).sum() / (abundance_weight.sum() * output["c_hat"].shape[1]).clamp(min=1.0)

        token_hidden = ~batch["token_visible"].reshape(batch["token_visible"].shape[0], -1)
        l_token = self._masked_mse(
            output["token_hat"], batch["token_raw"], token_hidden[..., None]
        )
        l_delta = (
            output["rho"] * output["delta_logits"].abs()
        ).mean()

        zero = output["od_hat"].new_zeros(())
        l_feature = zero
        if output_b is not None:
            a = F.normalize(output["features"].flatten(2), dim=1)
            b = F.normalize(output_b["features"].flatten(2), dim=1)
            l_feature = 0.5 * (
                (1.0 - (a * b.detach()).sum(dim=1)).mean()
                + (1.0 - (b * a.detach()).sum(dim=1)).mean()
            )

        l_sam = zero
        if self.weights["sam"] > 0:
            pred = output["od_hat"].flatten(2)
            target = batch["od"].flatten(2)
            cosine = F.cosine_similarity(pred, target, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
            pixel_valid = valid.flatten(2).squeeze(1)
            l_sam = (torch.acos(cosine) * pixel_valid).sum() / pixel_valid.sum().clamp(min=1)

        values = {
            "od": l_od, "i": l_i, "c": l_c, "token": l_token,
            "feature": l_feature, "delta": l_delta, "sam": l_sam,
        }
        total = sum(self.weights[key] * value for key, value in values.items())
        logs = {f"loss_{key}": float(value.detach()) for key, value in values.items()}
        logs["loss_total"] = float(total.detach())
        return total, logs
