from pathlib import Path

import numpy as np
import pytest
import torch

from models.endmember_conditioned_pretrain_model import EndmemberConditionedPretrainModel
from models.finetune_model_conditioned_cls import (
    CLASSIFICATION_HEAD_TYPES,
    ConditionedClassificationModel,
)
from split_pretrain_finetune import (
    Bucket,
    Item,
    classification_group_id,
    split_classification_grouped_items,
)
from tests.conditioned_test_utils import make_batch, tiny_config
from utils.classification_metrics import compute_classification_metrics
from utils.datasets.conditioned_classification_dataset import (
    ConditionedClassificationDataset,
    classification_collate,
)
from utils.preprocessing.offline_nmf import cache_dir_name


def _write_classification_root(root: Path) -> None:
    rng = np.random.default_rng(4)
    nmf_name = cache_dir_name(3, 5e-4, 2e-4, 1e-2, True, 0.05, 3.0)
    np.save(root / "wavelengths.npy", np.linspace(450.0, 700.0, 8, dtype=np.float32))
    for class_name in ("A", "B"):
        image_dir = root / class_name / "images"
        nmf_dir = root / class_name / nmf_name
        image_dir.mkdir(parents=True)
        nmf_dir.mkdir(parents=True)
        for index in range(2):
            stem = f"{class_name}-roi{index}_p0"
            intensity = rng.uniform(0.2, 1.0, size=(16, 16, 8)).astype(np.float32)
            endmembers = rng.uniform(0.1, 1.0, size=(3, 8)).astype(np.float32)
            np.save(image_dir / f"{stem}.npy", intensity)
            np.save(nmf_dir / f"{stem}_E.npy", endmembers)


def test_classification_dataset_contract(tmp_path):
    _write_classification_root(tmp_path)
    dataset = ConditionedClassificationDataset(
        str(tmp_path),
        patch_size=4,
        spectral_patch_size=2,
        nmf_k=3,
        augment=True,
    )
    assert dataset.class_to_idx == {"A": 0, "B": 1}
    assert dataset.class_counts().tolist() == [2, 2]
    sample = dataset[0]
    assert sample["od"].shape == (8, 16, 16)
    assert sample["e_star"].shape == (3, 8)
    assert sample["token_raw"].shape == (64, 32)
    assert sample["label"].dtype == torch.long
    batch = classification_collate([dataset[0], dataset[2]])
    assert batch["od"].shape == (2, 8, 16, 16)
    assert batch["label"].tolist() == [0, 1]


def test_classification_online_augmentation_copies_are_distinct(tmp_path):
    _write_classification_root(tmp_path)
    dataset = ConditionedClassificationDataset(
        str(tmp_path),
        patch_size=4,
        spectral_patch_size=2,
        nmf_k=3,
        augment=True,
        augmentation_copies=4,
        augmentation_seed=17,
    )
    assert len(dataset.samples) == 4
    assert len(dataset) == 16
    assert dataset.class_counts().tolist() == [8, 8]

    dataset.set_epoch(3)
    views = [dataset[index] for index in range(4)]
    transform_ids = [int(view["augmentation_transform_id"]) for view in views]
    assert len(set(transform_ids)) == 4
    assert [int(view["base_sample_index"]) for view in views] == [0, 0, 0, 0]
    assert [int(view["augmentation_copy_index"]) for view in views] == [0, 1, 2, 3]

    # Transform assignment is deterministic inside an epoch, including when
    # persistent DataLoader workers revisit the same virtual index.
    assert int(dataset[0]["augmentation_transform_id"]) == transform_ids[0]


def test_classification_augmentation_copy_validation(tmp_path):
    _write_classification_root(tmp_path)
    common = dict(
        data_root=str(tmp_path),
        patch_size=4,
        spectral_patch_size=2,
        nmf_k=3,
    )
    with pytest.raises(ValueError, match="augmentation_copies"):
        ConditionedClassificationDataset(
            **common, augment=True, augmentation_copies=9
        )
    with pytest.raises(ValueError, match="augment=False"):
        ConditionedClassificationDataset(
            **common, augment=False, augmentation_copies=2
        )


def test_conditioned_classification_forward_and_freeze():
    model = ConditionedClassificationModel(5, tiny_config(), head_dropout=0.0)
    model.eval()
    with torch.no_grad():
        logits, output = model.forward_with_features(make_batch())
    assert logits.shape == (1, 5)
    assert output["features"].shape == (1, 16, 16, 16)
    assert not any(
        parameter.requires_grad for parameter in model.backbone.abundance_head.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.backbone.token_reconstruction_head.parameters()
    )

    frozen = ConditionedClassificationModel(
        5, tiny_config(), freeze_backbone=True, head_dropout=0.0
    )
    frozen.train()
    assert not frozen.backbone.training
    assert frozen.cls_head.training
    assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())


def test_conditioned_classification_backward():
    model = ConditionedClassificationModel(3, tiny_config(), head_dropout=0.0)
    logits = model(make_batch())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    loss.backward()
    assert model.cls_head.classifier.weight.grad is not None
    assert model.backbone.feature_decoder.output.weight.grad is not None
    assert model.backbone.abundance_head.feature_projection.weight.grad is None


@pytest.mark.parametrize("head_type", CLASSIFICATION_HEAD_TYPES)
def test_conditioned_classification_head_variants_forward_backward(head_type):
    model = ConditionedClassificationModel(
        3,
        tiny_config(),
        head_dropout=0.0,
        classification_head=head_type,
        head_projection_dim=8,
        head_hidden_dim=12,
    )
    logits, output = model.forward_with_features(make_batch())
    assert logits.shape == (1, 3)
    assert ("decoder_stages" in output) == (head_type == "h3_multiscale_gated")
    logits.sum().backward()
    assert any(
        parameter.grad is not None for parameter in model.cls_head.parameters()
    )


def test_conditioned_classification_rejects_unknown_head():
    with pytest.raises(ValueError, match="unknown classification head"):
        ConditionedClassificationModel(
            3, tiny_config(), classification_head="not_a_head"
        )


def test_conditioned_classification_loads_pretrain_checkpoint(tmp_path):
    backbone = EndmemberConditionedPretrainModel(tiny_config())
    path = tmp_path / "pretrain.pth"
    torch.save({"model": backbone.state_dict()}, path)
    model = ConditionedClassificationModel(3, tiny_config())
    summary = model.load_pretrain(str(path))
    assert summary["matched"] == len(backbone.state_dict())
    assert summary["shape_mismatch"] == 0


def test_classification_metrics_perfect_predictions():
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.full((targets.numel(), 3), -4.0)
    logits[torch.arange(targets.numel()), targets] = 4.0
    metrics = compute_classification_metrics(logits, targets, 3)
    assert metrics["Accuracy"] == 1.0
    assert metrics["MacroF1"] == 1.0
    assert metrics["MacroAUC"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_classification_metrics_macro_f1_and_tied_auc():
    targets = np.asarray([0, 0, 1, 1])
    logits = np.zeros((4, 2), dtype=np.float32)
    metrics = compute_classification_metrics(logits, targets, 2)
    # All predictions are class 0: class F1 values are 2/3 and 0.
    assert np.isclose(metrics["Accuracy"], 0.5)
    assert np.isclose(metrics["MacroF1"], 1.0 / 3.0)
    assert np.isclose(metrics["MacroAUC"], 0.5)


def _item(stem: str) -> Item:
    root = Path("/tmp/classification-test")
    bucket = Bucket("A", root, root / "images", None, root / "nmf")
    return Item(bucket, stem, 0.0, True)


def test_group_aware_classification_split_keeps_rois_together():
    items = [
        _item(f"A-roi{roi}_p{patch}")
        for roi in range(10)
        for patch in range(3)
    ]
    regex = r"^(.+)_p.*$"
    splits = split_classification_grouped_items(items, 0.3, 0.2, 0.2, 7, regex)
    seen: dict[str, str] = {}
    for split_name, split_items in splits.items():
        for item in split_items:
            group_id = classification_group_id(item.stem, regex)
            assert group_id not in seen or seen[group_id] == split_name
            seen[group_id] = split_name
    assert len(seen) == 10
    assert sum(len(values) for values in splits.values()) == len(items)
