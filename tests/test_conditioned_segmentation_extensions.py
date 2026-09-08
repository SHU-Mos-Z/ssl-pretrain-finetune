from __future__ import annotations

import numpy as np
import torch

from models.finetune_model_vit import SegmentationHead
from models.modules_conditioned.segmentation_heads import build_segmentation_head
from utils.augmentations.segmentation_spatial import augment_hsi_segmentation_pair
from utils.losses import build_segmentation_criterion, primary_segmentation_logits
from utils.losses.soft_dice_ce_loss import SoftDiceCrossEntropyLoss


def test_historical_head_and_loss_are_exactly_preserved() -> None:
    torch.manual_seed(7)
    historical = SegmentationHead(16, 4)
    configured = build_segmentation_head(
        "h0_simple", feature_channels=16, decoder_channels=8, num_classes=4
    )
    configured.load_state_dict(historical.state_dict(), strict=True)
    historical.eval()
    configured.eval()
    features = torch.randn(2, 16, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    expected = historical(features)
    actual = configured(features)
    old_loss, _ = SoftDiceCrossEntropyLoss(4)(expected, target)
    new_loss, _ = build_segmentation_criterion(4)(actual, target)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    assert list(historical.state_dict()) == list(configured.state_dict())


def test_all_new_heads_and_losses_have_finite_gradients() -> None:
    features = torch.randn(2, 16, 32, 32)
    stages = {
        "D2": torch.randn(2, 8, 8, 8),
        "D1": torch.randn(2, 8, 16, 16),
    }
    target = torch.randint(0, 4, (2, 32, 32))
    for head_name in (
        "h1_residual",
        "h2_aspp",
        "h3_multiscale_aux",
    ):
        head = build_segmentation_head(
            head_name,
            feature_channels=16,
            decoder_channels=8,
            num_classes=4,
            hidden_channels=16,
            projection_channels=8,
            aspp_rates=(1, 2, 3),
        )
        prediction = (
            head(features, stages) if head_name == "h3_multiscale_aux" else head(features)
        )
        criterion = build_segmentation_criterion(
            4,
            loss_type="weighted_ce_dice_boundary",
            class_weights=torch.tensor([0.6, 0.8, 1.2, 1.4]),
            boundary_weight=0.2,
            auxiliary_weight=0.4 if head_name == "h3_multiscale_aux" else 0.0,
        )
        loss, _ = criterion(prediction, target)
        loss.backward()
        assert tuple(primary_segmentation_logits(prediction).shape) == (2, 4, 32, 32)
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        )


def test_dihedral_augmentation_keeps_hsi_and_mask_aligned() -> None:
    mask = np.arange(64, dtype=np.int64).reshape(8, 8) % 4
    intensity = np.stack([(mask.astype(np.float32) + 1.0) / 5.0] * 5)
    result = augment_hsi_segmentation_pair(
        intensity,
        mask,
        policy="dihedral",
        base_seed=123,
        epoch=4,
        sample_index=2,
        copy_index=0,
    )
    recovered = np.rint(result.intensity[0] * 5.0 - 1.0).astype(np.int64)
    np.testing.assert_array_equal(recovered, result.mask)
    assert set(np.unique(result.mask)).issubset({0, 1, 2, 3})
