from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.finetune_model_vit import SegmentationHead
from models.modules_conditioned.segmentation_heads import build_segmentation_head
from utils.augmentations.segmentation_spatial import augment_hsi_segmentation_pair
from utils.losses import build_segmentation_criterion, primary_segmentation_logits
from utils.losses.soft_dice_ce_loss import SoftDiceCrossEntropyLoss
from utils.datasets.conditioned_finetune_dataset import (
    ConditionedSlidingWindowSceneDataset,
    build_conditioned_finetune_loaders,
)
from utils.preprocessing.offline_nmf import cache_dir_name


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


def test_foreground_loss_mode_uses_weighted_all_class_ce_and_foreground_dice() -> None:
    logits = torch.tensor(
        [[
            [[3.0, 0.2], [0.1, 0.3]],
            [[0.1, 2.5], [1.5, 0.2]],
            [[0.0, 0.1], [0.2, 2.0]],
        ]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1], [1, 2]]])
    weights = torch.tensor([0.5, 1.0, 1.5])
    criterion = build_segmentation_criterion(
        3,
        loss_type="ce_dice",
        class_weights=weights,
        computation_mode="foreground",
    )
    loss, logs = criterion(logits, target)

    expected_ce = F.cross_entropy(logits, target, weight=weights)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, 3).permute(0, 3, 1, 2).to(logits.dtype)
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = (probabilities + one_hot).sum((0, 2, 3))
    foreground_dice_loss = 1.0 - (
        (2.0 * intersection[1:] + 1e-6) / (denominator[1:] + 1e-6)
    ).mean()

    torch.testing.assert_close(
        torch.tensor(logs["loss_ce_or_focal"]), expected_ce.detach()
    )
    torch.testing.assert_close(
        torch.tensor(logs["loss_dice"]), foreground_dice_loss.detach()
    )
    assert criterion.boundary_weight == 0.2
    loss.backward()
    assert torch.isfinite(logits.grad).all()


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


def test_optional_complete_scene_validation_preserves_legacy_loader_path(tmp_path) -> None:
    cache_name = cache_dir_name(2, 5e-4, 2e-4, 1e-2, True, 0.05, 3.0)
    for split, shape in (
        ("train", (16, 16, 5)),
        ("val", (16, 16, 5)),
        ("val_scenes", (80, 84, 5)),
        ("test", (80, 84, 5)),
    ):
        root = tmp_path / split
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir()
        (root / cache_name).mkdir()
        np.save(root / "wavelengths.npy", np.arange(5, dtype=np.float32))
        np.save(
            root / "images" / "subject-1.npy",
            np.full(shape, 0.5, dtype=np.float32),
        )
        np.save(
            root / "masks" / "subject-1.npy",
            np.zeros(shape[:2], dtype=np.uint8),
        )
        np.save(
            root / cache_name / "subject-1_E.npy",
            np.ones((2, 5), dtype=np.float32),
        )

    shared = dict(
        batch_size=1,
        num_workers=0,
        patch_size=4,
        spectral_patch_size=5,
        nmf_k=2,
        nmf_simplex=True,
        nmf_e_clamp_max=3.0,
    )
    train, val, scene_val, test, sampler = build_conditioned_finetune_loaders(
        str(tmp_path / "train"),
        str(tmp_path / "val"),
        str(tmp_path / "test"),
        test_inference_mode="sliding_window",
        scene_val_root=str(tmp_path / "val_scenes"),
        **shared,
    )
    assert len(train.dataset) == 1
    assert len(val.dataset) == 1
    assert isinstance(scene_val, ConditionedSlidingWindowSceneDataset)
    assert isinstance(test, ConditionedSlidingWindowSceneDataset)
    assert sampler is None

    legacy_result = build_conditioned_finetune_loaders(
        str(tmp_path / "train"),
        str(tmp_path / "val"),
        None,
        **shared,
    )
    assert len(legacy_result) == 4
    _, _, legacy_test, _ = legacy_result
    assert legacy_test is None
