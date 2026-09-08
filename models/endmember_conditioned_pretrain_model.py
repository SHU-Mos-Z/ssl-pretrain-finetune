"""Endmember-conditioned residual-abundance pretraining model."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from models.conditioned_contracts import ConditionedModelConfig
from models.modules_conditioned import (
    ConditionedTokenEncoder,
    ConfidenceGatedFusion,
    EndmemberSetEncoder,
    GatedFeatureDecoder,
    MaskedContextEncoder,
    PhysicsPriorFusion,
    ResidualAbundanceHead,
    TokenReconstructionHead,
    VisibilitySpectralAggregate,
)
from models.modules_vit.physics_decode import reconstruct_intensity, reconstruct_od
from models.modules_vit.vit_backbone import ViTBackbone
from utils.physics.coarse_unmixing import coarse_unmix


class EndmemberConditionedPretrainModel(nn.Module):
    """The stage-2 model path described in the conditioned pretraining method."""

    def __init__(self, config: ConditionedModelConfig | None = None):
        super().__init__()
        self.config = config or ConditionedModelConfig()
        self.config.validate()
        cfg = self.config
        spatial_depth = int(math.log2(cfg.patch_size))

        self.endmember_encoder = EndmemberSetEncoder(
            cfg.spectral_patch_size, cfg.embed_dim
        )
        self.token_encoder = ConditionedTokenEncoder(
            cfg.patch_size,
            cfg.spectral_patch_size,
            cfg.embed_dim,
            cfg.vit_heads,
            cfg.dropout,
        )
        self.prior_fusion = PhysicsPriorFusion(cfg.embed_dim)
        self.vit = ViTBackbone(
            cfg.embed_dim,
            cfg.vit_depth,
            cfg.vit_heads,
            cfg.mlp_ratio,
            cfg.dropout,
        )
        self.context_encoder = MaskedContextEncoder(
            cfg.cnn_stem_ch, spatial_depth, cfg.cnn_spectral_agg
        )
        self.confidence_fusion = ConfidenceGatedFusion(
            self.context_encoder.out_ch,
            cfg.embed_dim,
            cfg.fusion_heads,
            cfg.dropout,
        )
        self.spectral_aggregate = VisibilitySpectralAggregate(cfg.embed_dim)
        self.feature_decoder = GatedFeatureDecoder(
            cfg.embed_dim,
            cfg.cnn_stem_ch,
            self.context_encoder.layer_channels,
            cfg.decoder_mid_ch,
            cfg.feature_dim,
        )
        self.abundance_head = ResidualAbundanceHead(
            cfg.feature_dim, cfg.embed_dim, cfg.residual_hidden_dim
        )
        self.token_reconstruction_head = TokenReconstructionHead(
            cfg.embed_dim,
            cfg.patch_size * cfg.patch_size * cfg.spectral_patch_size,
        )

    def forward_features(
        self,
        batch: dict[str, Any],
        return_decoder_stages: bool = False,
        decoder_stop_at_stage: int = 0,
    ) -> dict[str, Any]:
        cfg = self.config
        od = batch["od"]
        e_star = batch["e_star"]
        wavelengths = batch["wavelengths"]
        token_raw = batch["token_raw"]
        token_visible = batch["token_visible"]
        voxel_visible = batch["voxel_visible"]
        pe_spatial = batch["pe_spatial"]
        pe_spectral = batch["pe_spectral"]

        if od.ndim != 4:
            raise ValueError("od must have shape (B,S,H,W)")
        b, s, h, w = od.shape
        if h % cfg.patch_size or w % cfg.patch_size:
            raise ValueError("H and W must be divisible by patch_size")
        if s % cfg.spectral_patch_size:
            raise ValueError("S must be divisible by spectral_patch_size")
        h_p, w_p = h // cfg.patch_size, w // cfg.patch_size
        num_groups = s // cfg.spectral_patch_size
        if token_visible.shape == (b, h_p * w_p * num_groups):
            visible_grid = token_visible.view(b, h_p, w_p, num_groups)
        elif token_visible.shape == (b, h_p, w_p, num_groups):
            visible_grid = token_visible
        else:
            raise ValueError("token_visible has an incompatible shape")

        coarse = coarse_unmix(
            od,
            e_star,
            voxel_visible,
            cfg.patch_size,
            cfg.ridge_lambda,
            cfg.confidence_temperature,
        )
        e_groups, e_tokens = self.endmember_encoder(e_star, wavelengths)
        x_embed, e_evidence = self.token_encoder(
            token_raw,
            visible_grid,
            pe_spatial,
            pe_spectral,
            e_groups,
            h_p,
            w_p,
        )
        x_conditioned = self.prior_fusion(
            x_embed,
            e_evidence,
            e_tokens,
            coarse.c0_low,
            coarse.rho_low,
            num_groups,
        )
        z_vit = self.vit(x_conditioned)

        e_cnn, skips = self.context_encoder(od, coarse.x0, voxel_visible, coarse.rho)
        z_fused, cnn_semantic = self.confidence_fusion(
            z_vit, e_cnn, coarse.rho_low, num_groups
        )
        z_grid = z_fused.view(b, h_p, w_p, num_groups, cfg.embed_dim)
        f_low = self.spectral_aggregate(z_grid, visible_grid)
        f_low = f_low + cnn_semantic.view(b, h_p, w_p, cfg.embed_dim)
        decoded = self.feature_decoder(
            f_low.permute(0, 3, 1, 2),
            skips,
            coarse.rho,
            return_intermediate=return_decoder_stages,
            stop_at_stage=decoder_stop_at_stage,
        )
        if return_decoder_stages:
            features, decoder_stages = decoded
        else:
            features = decoded
            decoder_stages = None
        output: dict[str, Any] = {
            "features": features,
            "c0_low": coarse.c0_low,
            "c0": coarse.c0,
            "x0": coarse.x0,
            "rho_low": coarse.rho_low,
            "rho": coarse.rho,
            "endmember_group_tokens": e_groups,
            "endmember_tokens": e_tokens,
            "x_embed": x_embed,
            "x_conditioned": x_conditioned,
            "z_vit": z_vit,
            "z_fused": z_fused,
            "f_low": f_low,
        }
        if decoder_stages is not None:
            output["decoder_stages"] = decoder_stages
        return output

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = self.forward_features(batch)
        cfg = self.config
        delta_logits, c_hat = self.abundance_head(
            output["features"],
            output["endmember_tokens"],
            output["c0"],
            output["rho"],
            cfg.alpha_min,
            cfg.alpha_extra,
        )
        od_hat = reconstruct_od(c_hat, batch["e_star"])
        i_hat = reconstruct_intensity(od_hat, cfg.od_max)
        token_hat = self.token_reconstruction_head(output["z_fused"])
        output.update(
            delta_logits=delta_logits,
            c_hat=c_hat,
            od_hat=od_hat,
            i_hat=i_hat,
            token_hat=token_hat,
        )
        return output
