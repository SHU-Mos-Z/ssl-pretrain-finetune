from __future__ import annotations

import json
from pathlib import Path
import math

import numpy as np
import pytest
import torch

from models.conditioned_contracts import ConditionedModelConfig
from models.detection_contracts import DetectionConfig
from models.finetune_model_conditioned_detection import ConditionedDetectionModel
from models.modules_detection import BoxCoder
from split_pretrain_finetune import Bucket, Item, split_detection_grouped_items
from utils.datasets.conditioned_detection_dataset import (
    ConditionedDetectionDataset,
    conditioned_detection_collate,
)
from utils.detection_metrics import evaluate_coco_detections
from utils.detection_postprocess import DetectionPostProcessor
from utils.detection_runtime import merge_source_detections
from utils.datasets.detection_view_geometry import (
    DetectionView,
    DetectionViewConfig,
    build_evaluation_views,
    inverse_project_boxes,
    project_annotations_to_view,
)
from utils.losses import DetectionCriterion
from utils.preprocessing.offline_nmf import cache_dir_name


def _model_config() -> ConditionedModelConfig:
    return ConditionedModelConfig(
        patch_size=16,
        spectral_patch_size=2,
        embed_dim=16,
        vit_depth=1,
        vit_heads=4,
        mlp_ratio=2,
        dropout=0,
        cnn_stem_ch=8,
        cnn_spectral_agg="mean",
        fusion_heads=4,
        feature_dim=8,
        decoder_mid_ch=8,
        residual_hidden_dim=8,
    )


def _batch(height: int, width: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(height + width)
    b, s, k = 1, 4, 3
    hp, wp, groups = height // 16, width // 16, s // 2
    tokens = hp * wp * groups
    return {
        "od": torch.rand(b, s, height, width),
        "intensity": torch.rand(b, s, height, width),
        "e_star": torch.rand(b, k, s),
        "wavelengths": torch.arange(s).float().repeat(b, 1),
        "token_raw": torch.rand(b, tokens, 16 * 16 * 2),
        "token_visible": torch.ones(b, hp, wp, groups, dtype=torch.bool),
        "voxel_visible": torch.ones(b, s, height, width, dtype=torch.bool),
        "pe_spatial": torch.rand(b, tokens, 2),
        "pe_spectral": torch.rand(b, tokens, 1),
    }


def _target(height: int, width: int) -> dict[str, torch.Tensor]:
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[:4, :4] = True
    return {
        "boxes": torch.tensor([[8.0, 8.0, min(30.0, width), min(30.0, height)]]),
        "labels": torch.tensor([0]),
        "crowd_boxes": torch.empty(0, 4),
        "ignore_boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "ignore_mask": mask,
    }


def test_box_coder_round_trip():
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 3.0, 20.0, 12.0]])
    boxes = torch.tensor([[2.0, 1.0, 13.0, 9.0], [1.0, 4.0, 18.0, 15.0]])
    coder = BoxCoder((1.0, 1.0, 1.0, 1.0))
    assert torch.allclose(coder.decode(anchors, coder.encode(anchors, boxes)), boxes, atol=1e-5)


@pytest.mark.parametrize("detection_mode", ["anchor_based", "anchor_free"])
@pytest.mark.parametrize("feature_mode", ["z_full", "z_pyramid", "gated_pyramid"])
@pytest.mark.parametrize("image_size", [(64, 80), (64, 64)])
def test_all_detection_modes_are_dynamic_and_differentiable(
    detection_mode: str, feature_mode: str, image_size: tuple[int, int]
):
    config = DetectionConfig(
        detection_mode=detection_mode,
        feature_mode=feature_mode,
        num_classes=1,
        det_feature_dim=8,
        head_depth=1,
        anchor_sizes=(8, 16, 32, 64),
        anchor_scales=(1.0,),
        anchor_ratios=(1.0,),
        pre_nms_topk=20,
        max_detections=10,
    )
    model = ConditionedDetectionModel(_model_config(), config)
    height, width = image_size
    output = model(_batch(height, width))
    losses = DetectionCriterion(config)(output, [_target(height, width)])
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    detections = DetectionPostProcessor(config)(output)
    assert detections[0]["boxes"].shape[1:] == (4,)
    expected = ((height, width),) if feature_mode == "z_full" else (
        (height // 4, width // 4),
        (height // 8, width // 8),
        (height // 16, width // 16),
        (math.ceil(height / 32), math.ceil(width / 32)),
    )
    assert output["feature_shapes"] == expected


@pytest.mark.parametrize("annotation_layout", ("monolithic", "fragments"))
def test_dataset_contract_and_rectangular_joint_augmentation(tmp_path, annotation_layout):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "ignore_masks").mkdir()
    (root / "annotations").mkdir()
    cache = root / cache_dir_name(3, 5e-4, 2e-4, 1e-2, True, 0.05, 3.0)
    cache.mkdir()
    np.save(root / "images" / "sample.npy", np.full((64, 80, 4), 0.75, np.float32))
    ignore = np.zeros((64, 80), np.uint8)
    ignore[0:5, 0:8] = 1
    np.save(root / "ignore_masks" / "sample.npy", ignore)
    np.save(cache / "sample_E.npy", np.full((3, 4), 0.2, np.float32))
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "images/sample.npy",
                "ignore_mask_file_name": "ignore_masks/sample.npy",
                "height": 64,
                "width": 80,
            }
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 8, 20, 16], "area": 320, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 7, "bbox": [0, 0, 8, 5], "area": 40, "iscrowd": 0, "ignore": 1},
        ],
        "categories": [{"id": 7, "name": "lesion"}],
    }
    annotation = root / "annotations" / (
        "sample.json" if annotation_layout == "fragments" else "instances.json"
    )
    annotation.write_text(json.dumps(coco), encoding="utf-8")
    annotation_source = annotation.parent if annotation_layout == "fragments" else annotation
    dataset = ConditionedDetectionDataset(
        root,
        annotation_source,
        patch_size=16,
        spectral_patch_size=2,
        nmf_k=3,
        allow_index_wavelengths=True,
        augment=True,
        augmentation_probability=1.0,
        seed=4,
    )
    model_inputs, target = dataset[0]
    assert model_inputs["od"].shape == (4, 64, 80)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [0]
    assert target["ignore_boxes"].shape == (1, 4)
    assert int(target["ignore_mask"].sum()) == 40
    batch, targets = conditioned_detection_collate([(model_inputs, target)])
    assert batch["od"].shape == (1, 4, 64, 80)
    assert len(targets) == 1


def test_coco_metrics_accept_custom_ignore(tmp_path):
    coco = {
        "images": [{"id": 1, "width": 64, "height": 64, "file_name": "x.npy"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "area": 100, "iscrowd": 0, "ignore": 1},
        ],
        "categories": [{"id": 1, "name": "lesion"}],
    }
    detections = [
        {
            "image_id": 1,
            "boxes": [[10, 10, 30, 30], [40, 40, 50, 50]],
            "scores": [0.9, 0.8],
            "labels": [0, 0],
        }
    ]
    pr_curve_path = tmp_path / "pr_curve.png"
    metrics = evaluate_coco_detections(
        coco, detections, {0: 1}, pr_curve_path=pr_curve_path
    )
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["AP75"] == pytest.approx(1.0)
    assert metrics["AP50_95"] == pytest.approx(1.0)
    assert pr_curve_path.is_file()
    assert pr_curve_path.with_suffix(".csv").is_file()


def test_group_aware_detection_split_keeps_sources_together():
    root = Path("/tmp/detection-split-test")
    bucket = Bucket(None, root, root / "images", root / "masks", root / "nmf")
    items = [
        Item(
            bucket,
            f"source{source:02d}_p{patch}",
            0.0,
            True,
            detection_group=f"source{source:02d}",
            ordinary_count=1 + (source % 3),
            category_counts={1: 1 + (source % 3)},
        )
        for source in range(12)
        for patch in range(2)
    ]
    splits = split_detection_grouped_items(items, 0.3, 0.2, 0.2, 17)
    group_to_split: dict[str, str] = {}
    for split_name, split_items in splits.items():
        assert split_items
        for item in split_items:
            group = str(item.detection_group)
            assert group not in group_to_split or group_to_split[group] == split_name
            group_to_split[group] = split_name
    assert len(group_to_split) == 12
    assert sum(len(split_items) for split_items in splits.values()) == len(items)


def test_runtime_supervision_policy_distinguishes_both_truncation_sources():
    config = DetectionViewConfig(
        view_mode="runtime_window",
        source_crop_size=(100, 100),
        model_input_size=(100, 100),
        eval_stride=(50, 50),
        visible_ratio_threshold=0.7,
        min_visible_side=10,
    )
    view = DetectionView(0, 1, "sample", (0, 0, 100, 100), (100, 100), (0, 0, 100, 100))
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [70, 20, 40, 40]},
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [80, 60, 40, 40]},
        {"id": 4, "image_id": 1, "category_id": 1, "bbox": [30, 50, 20, 20], "truncated": 1},
        {"id": 5, "image_id": 1, "category_id": 1, "bbox": [90, 5, 20, 20], "truncated": 1},
    ]
    projected = project_annotations_to_view(annotations, view, config)
    positive = {item["id"]: item for item in projected["positive"]}
    ignored = {item["id"]: item for item in projected["ignored"]}
    assert set(positive) == {1, 2, 4}
    assert positive[2]["crop_truncated"] and not positive[2]["source_truncated"]
    assert positive[4]["source_truncated"] and not positive[4]["crop_truncated"]
    assert set(ignored) == {3, 5}
    assert not set(positive).intersection(ignored)


def test_evaluation_views_cover_source_with_unique_ownership_and_round_trip():
    config = DetectionViewConfig(
        view_mode="runtime_window",
        source_crop_size=(64, 64),
        model_input_size=(32, 32),
        eval_stride=(31, 29),
    )
    views = build_evaluation_views(
        source_image_id=7,
        source_stem="scene",
        source_size=(130, 180),
        config=config,
    )
    assert max(view.crop_xyxy[2] for view in views) == 180
    assert max(view.crop_xyxy[3] for view in views) == 130
    for y in np.arange(0.5, 130, 7):
        for x in np.arange(0.5, 180, 7):
            owners = [
                view
                for view in views
                if view.ownership_xyxy[0] <= x < view.ownership_xyxy[2]
                and view.ownership_xyxy[1] <= y < view.ownership_xyxy[3]
            ]
            assert len(owners) == 1
    view = views[len(views) // 2]
    source_box = np.asarray([[view.crop_xyxy[0] + 4, view.crop_xyxy[1] + 8,
                              view.crop_xyxy[0] + 40, view.crop_xyxy[1] + 52]], np.float32)
    scale_x, scale_y = view.scale_xy
    projected = source_box.copy()
    projected[:, 0::2] = (projected[:, 0::2] - view.crop_xyxy[0]) * scale_x
    projected[:, 1::2] = (projected[:, 1::2] - view.crop_xyxy[1]) * scale_y
    assert np.allclose(inverse_project_boxes(projected, view), source_box, atol=1e-5)


def test_source_level_global_nms_merges_overlapping_view_predictions():
    merged = merge_source_detections(
        [{
            "image_id": 1,
            "boxes": [[10, 10, 30, 30], [11, 11, 31, 31], [60, 60, 80, 80]],
            "scores": [0.9, 0.8, 0.7],
            "labels": [0, 0, 0],
        }],
        nms_threshold=0.5,
        max_detections=10,
    )
    assert len(merged) == 1
    assert len(merged[0]["boxes"]) == 2


def test_runtime_window_dataset_crops_before_tokenization(tmp_path):
    root = tmp_path / "runtime"
    (root / "images").mkdir(parents=True)
    (root / "ignore_masks").mkdir()
    (root / "annotations").mkdir()
    cache = root / cache_dir_name(3, 5e-4, 2e-4, 1e-2, True, 0.05, 3.0)
    cache.mkdir()
    np.save(root / "images" / "scene.npy", np.full((96, 112, 4), 0.75, np.float32))
    np.save(root / "ignore_masks" / "scene.npy", np.zeros((96, 112), np.uint8))
    np.save(cache / "scene_E.npy", np.full((3, 4), 0.2, np.float32))
    coco = {
        "images": [{"id": 9, "file_name": "images/scene.npy",
                    "ignore_mask_file_name": "ignore_masks/scene.npy",
                    "height": 96, "width": 112}],
        "annotations": [{"id": 1, "image_id": 9, "category_id": 1,
                         "bbox": [30, 20, 40, 40], "area": 1600, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "WBC"}],
    }
    (root / "annotations" / "scene.json").write_text(json.dumps(coco), encoding="utf-8")
    view_config = DetectionViewConfig(
        view_mode="runtime_window", source_crop_size=(64, 64),
        model_input_size=(32, 32), eval_stride=(32, 32),
        train_views_per_source=2, positive_guided_fraction=1.0, seed=3,
    )
    dataset = ConditionedDetectionDataset(
        root, "annotations", patch_size=16, spectral_patch_size=2,
        nmf_k=3, allow_index_wavelengths=True, view_config=view_config,
        training=True, augment=False,
    )
    assert len(dataset) == 2
    inputs, target = dataset[0]
    assert inputs["od"].shape == (4, 32, 32)
    assert target["image_size"].tolist() == [32, 32]
    assert int(target["source_image_id"].item()) == 9
    assert target["boxes"].shape == (1, 4)


@pytest.mark.parametrize(
    "crop_size,detection_mode",
    [((80, 80), "anchor_free"), ((80, 80), "anchor_based"), ((64, 64), "anchor_free")],
)
def test_three_runtime_input_head_combinations_smoke(tmp_path, crop_size, detection_mode):
    root = tmp_path / f"{crop_size[0]}_{detection_mode}"
    (root / "images").mkdir(parents=True)
    (root / "ignore_masks").mkdir()
    (root / "annotations").mkdir()
    cache = root / cache_dir_name(3, 5e-4, 2e-4, 1e-2, True, 0.05, 3.0)
    cache.mkdir()
    np.save(root / "images" / "scene.npy", np.full((96, 112, 4), 0.75, np.float32))
    np.save(root / "ignore_masks" / "scene.npy", np.zeros((96, 112), np.uint8))
    np.save(cache / "scene_E.npy", np.full((3, 4), 0.2, np.float32))
    coco = {
        "images": [{"id": 1, "file_name": "images/scene.npy",
                    "ignore_mask_file_name": "ignore_masks/scene.npy",
                    "height": 96, "width": 112}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [32, 24, 36, 36], "area": 1296, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "WBC"}],
    }
    (root / "annotations" / "scene.json").write_text(json.dumps(coco), encoding="utf-8")
    view_config = DetectionViewConfig(
        view_mode="runtime_window", source_crop_size=crop_size,
        model_input_size=(64, 64), eval_stride=(32, 32),
        train_views_per_source=1, positive_guided_fraction=1.0, seed=11,
    )
    dataset = ConditionedDetectionDataset(
        root, "annotations", patch_size=16, spectral_patch_size=2,
        nmf_k=3, allow_index_wavelengths=True, view_config=view_config,
        training=True, augment=False,
    )
    model_inputs, target = dataset[0]
    batch, targets = conditioned_detection_collate([(model_inputs, target)])
    config = DetectionConfig(
        detection_mode=detection_mode, feature_mode="gated_pyramid", num_classes=1,
        det_feature_dim=8, head_depth=1, anchor_sizes=(8, 16, 32, 64),
        anchor_scales=(1.0,), anchor_ratios=(1.0,), pre_nms_topk=20,
        max_detections=10,
    )
    model = ConditionedDetectionModel(_model_config(), config)
    output = model(batch)
    losses = DetectionCriterion(config)(output, targets)
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    prediction = DetectionPostProcessor(config)(output, [(64, 64)])[0]
    assert prediction["boxes"].shape[1:] == (4,)
