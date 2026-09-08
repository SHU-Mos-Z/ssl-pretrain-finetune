from __future__ import annotations

import torch
import torch.nn as nn

from models.conditioned_contracts import ConditionedModelConfig
from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from models.modules_conditioned.segmentation_heads import build_segmentation_head


class ConditionedFinetuneModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        config: ConditionedModelConfig,
        pretrain_ckpt=None,
        freeze_backbone=False,
        segmentation_head: str = "h0_simple",
        head_hidden_channels: int = 128,
        head_projection_channels: int = 64,
        head_dropout: float = 0.1,
        aspp_rates: tuple[int, ...] = (1, 6, 12, 18),
    ):
        super().__init__(); self.config=config
        self.backbone=EndmemberConditionedPretrainModel(config)
        self.segmentation_head_type = str(segmentation_head)
        self.seg_head=build_segmentation_head(
            self.segmentation_head_type,
            feature_channels=config.feature_dim,
            decoder_channels=config.decoder_mid_ch,
            num_classes=num_classes,
            hidden_channels=head_hidden_channels,
            projection_channels=head_projection_channels,
            dropout=head_dropout,
            aspp_rates=aspp_rates,
        )
        for module in (self.backbone.abundance_head,self.backbone.token_reconstruction_head):
            for parameter in module.parameters(): parameter.requires_grad_(False)
        if pretrain_ckpt: self.load_pretrain(pretrain_ckpt)
        if freeze_backbone:
            for parameter in self.backbone.parameters(): parameter.requires_grad_(False)

    def load_pretrain(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False); state=state.get('model',state)
        missing,unexpected=self.backbone.load_state_dict(state,strict=False)
        matched=len(state)-len(unexpected)
        if matched == 0: raise RuntimeError(f'no backbone weights matched {path}')
        print(f'[ConditionedFinetuneModel] matched={matched} missing={len(missing)} unexpected={len(unexpected)}')

    def forward_with_features(self,batch):
        multiscale = self.segmentation_head_type == "h3_multiscale_aux"
        output=self.backbone.forward_features(batch, return_decoder_stages=multiscale)
        if multiscale:
            prediction = self.seg_head(output['features'], output['decoder_stages'])
        else:
            prediction = self.seg_head(output['features'])
        return prediction,output

    def forward(self,batch): return self.forward_with_features(batch)[0]
