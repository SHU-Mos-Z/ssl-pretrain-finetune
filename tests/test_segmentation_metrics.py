from __future__ import annotations

import math

import pytest
import torch

from utils.metrics import (
    DICE_BATCH_ALLCLASS_MACRO,
    DICE_BATCH_FG_MACRO,
    DICE_CLASSWISE,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS,
    DICE_GLOBAL_FREQUENCY_WEIGHTED_FG,
    DICE_GLOBAL_FG,
    DICE_METRIC_NAMES,
    IOU_BATCH_ALLCLASS_MACRO,
    IOU_BATCH_FG_MACRO,
    IOU_CLASSWISE,
    IOU_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS,
    IOU_GLOBAL_FREQUENCY_WEIGHTED_FG,
    IOU_METRIC_NAMES,
    SegmentationMetricAccumulator,
    compute_segmentation_metrics,
    infer_segmentation_scene_id,
    normalize_dice_metric_names,
)


def _labels_to_logits(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = torch.full(
        (labels.shape[0], num_classes, *labels.shape[1:]),
        -8.0,
        dtype=torch.float32,
    )
    return logits.scatter_(1, labels.unsqueeze(1), 8.0)


def _historical_batch_dice(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> float:
    pred = logits.softmax(dim=1).argmax(dim=1)
    values = []
    for class_index in range(num_classes):
        pred_c = pred == class_index
        target_c = target == class_index
        intersection = (pred_c & target_c).sum().float()
        values.append(
            2.0
            * intersection
            / (pred_c.sum().float() + target_c.sum().float() + 1e-5)
        )
    return torch.stack(values).mean().item()


def test_default_metric_is_exact_historical_batch_macro() -> None:
    target = torch.tensor(
        [[[0, 1], [2, 0]], [[0, 0], [1, 2]]], dtype=torch.long
    )
    pred = torch.tensor(
        [[[0, 1], [0, 0]], [[0, 2], [1, 2]]], dtype=torch.long
    )
    logits = _labels_to_logits(pred, num_classes=3)
    metrics = compute_segmentation_metrics(logits, target, num_classes=3)

    assert tuple(metrics["Dice"]) == (DICE_BATCH_ALLCLASS_MACRO,)
    assert metrics["Dice"][DICE_BATCH_ALLCLASS_MACRO] == pytest.approx(
        _historical_batch_dice(logits, target, 3), abs=1e-7
    )


def test_metric_name_selection_ignores_invalid_and_falls_back() -> None:
    with pytest.warns(RuntimeWarning):
        assert normalize_dice_metric_names(["bad", DICE_GLOBAL_FG, "bad2"]) == (
            DICE_GLOBAL_FG,
        )
    with pytest.warns(RuntimeWarning):
        assert normalize_dice_metric_names(["bad"]) == (
            DICE_BATCH_ALLCLASS_MACRO,
        )
    assert normalize_dice_metric_names([]) == (DICE_BATCH_ALLCLASS_MACRO,)


def test_all_overlap_metrics_are_returned_with_classwise_mapping() -> None:
    target = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
    logits = _labels_to_logits(target.clone(), num_classes=3)
    metrics = compute_segmentation_metrics(
        logits,
        target,
        num_classes=3,
        dice_metrics=list(DICE_METRIC_NAMES),
        scene_ids=["scene-a"],
    )

    assert tuple(metrics["Dice"]) == DICE_METRIC_NAMES
    assert set(metrics["Dice"][DICE_CLASSWISE]) == {
        "class_0",
        "class_1",
        "class_2",
    }
    assert tuple(metrics["IoU_metrics"]) == IOU_METRIC_NAMES
    assert set(metrics["IoU_metrics"][IOU_CLASSWISE]) == {
        "class_0",
        "class_1",
        "class_2",
    }
    assert metrics["IoU"] == pytest.approx(
        metrics["IoU_metrics"][IOU_BATCH_ALLCLASS_MACRO]
    )
    for name, value in metrics["Dice"].items():
        if isinstance(value, dict):
            assert all(item == pytest.approx(1.0) for item in value.values())
        elif name == DICE_BATCH_ALLCLASS_MACRO:
            # Historical implementation adds epsilon only to the denominator.
            assert value == pytest.approx(1.0, abs=1e-5)
        else:
            assert value == pytest.approx(1.0)
    for name, value in metrics["IoU_metrics"].items():
        if isinstance(value, dict):
            assert all(item == pytest.approx(1.0) for item in value.values())
        elif name == IOU_BATCH_ALLCLASS_MACRO:
            assert value == pytest.approx(1.0, abs=1e-5)
        else:
            assert value == pytest.approx(1.0)


def test_global_and_scene_metrics_are_batch_partition_invariant() -> None:
    target = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, 2], [2, 0]],
            [[0, 1], [2, 0]],
        ],
        dtype=torch.long,
    )
    pred = torch.tensor(
        [
            [[0, 1], [0, 0]],
            [[0, 2], [1, 0]],
            [[0, 2], [2, 0]],
        ],
        dtype=torch.long,
    )
    batch_dependent = {DICE_BATCH_ALLCLASS_MACRO, DICE_BATCH_FG_MACRO}
    names = [name for name in DICE_METRIC_NAMES if name not in batch_dependent]

    whole = SegmentationMetricAccumulator(3, names)
    whole.update(pred, target, scene_ids=["s1", "s1", "s2"])
    whole_metrics = whole.compute()["Dice"]

    split = SegmentationMetricAccumulator(3, names)
    split.update(pred[:1], target[:1], scene_ids=["s1"])
    split.update(pred[1:], target[1:], scene_ids=["s1", "s2"])
    split_metrics = split.compute()["Dice"]

    for name in names:
        if name == DICE_CLASSWISE:
            assert split_metrics[name] == pytest.approx(whole_metrics[name])
        else:
            assert split_metrics[name] == pytest.approx(whole_metrics[name])

    whole_iou = whole.compute()["IoU_metrics"]
    split_iou = split.compute()["IoU_metrics"]
    for name in IOU_METRIC_NAMES:
        if name in {IOU_BATCH_ALLCLASS_MACRO, IOU_BATCH_FG_MACRO}:
            continue
        assert split_iou[name] == pytest.approx(whole_iou[name])


def test_batch_fg_macro_skips_jointly_absent_classes_and_excludes_background() -> None:
    # class 1 is perfect; class 2 is absent from both sides and must be skipped.
    target = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
    pred = target.clone()
    metrics = compute_segmentation_metrics(
        _labels_to_logits(pred, num_classes=3),
        target,
        num_classes=3,
        dice_metrics=[DICE_BATCH_FG_MACRO],
    )

    assert metrics["Dice"][DICE_BATCH_FG_MACRO] == pytest.approx(1.0)
    assert metrics["IoU_metrics"][IOU_BATCH_FG_MACRO] == pytest.approx(1.0)


def test_batch_fg_macro_counts_one_sided_presence_as_zero() -> None:
    # class 1 is perfect and class 2 is a pure false positive, so foreground
    # macro Dice/IoU are both (1 + 0) / 2.
    target = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
    pred = torch.tensor([[[2, 0], [1, 1]]], dtype=torch.long)
    metrics = compute_segmentation_metrics(
        _labels_to_logits(pred, num_classes=3),
        target,
        num_classes=3,
        dice_metrics=[DICE_BATCH_FG_MACRO],
    )

    assert metrics["Dice"][DICE_BATCH_FG_MACRO] == pytest.approx(0.5)
    assert metrics["IoU_metrics"][IOU_BATCH_FG_MACRO] == pytest.approx(0.5)


def test_batch_fg_macro_skips_fully_empty_batches_and_weights_valid_batches() -> None:
    accumulator = SegmentationMetricAccumulator(2, [DICE_BATCH_FG_MACRO])
    empty = torch.zeros((2, 2, 2), dtype=torch.long)
    accumulator.update(empty, empty)

    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    pred = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.long)
    accumulator.update(pred, target)
    metrics = accumulator.compute()

    assert metrics["Dice"][DICE_BATCH_FG_MACRO] == pytest.approx(2.0 / 3.0)
    assert metrics["IoU_metrics"][IOU_BATCH_FG_MACRO] == pytest.approx(0.5)


def test_global_frequency_weighted_dice_and_iou_use_gt_pixel_frequency() -> None:
    target = torch.tensor(
        [[[0, 0, 0, 1], [1, 2, 2, 2]]], dtype=torch.long
    )
    pred = torch.tensor(
        [[[0, 0, 1, 1], [1, 2, 0, 2]]], dtype=torch.long
    )
    metrics = compute_segmentation_metrics(
        _labels_to_logits(pred, num_classes=3),
        target,
        num_classes=3,
        dice_metrics=list(DICE_METRIC_NAMES),
        scene_ids=["scene-a"],
    )

    assert metrics["Dice"][DICE_GLOBAL_FREQUENCY_WEIGHTED_FG] == pytest.approx(0.8)
    assert metrics["Dice"][DICE_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS] == pytest.approx(
        0.75
    )
    assert metrics["IoU_metrics"][IOU_GLOBAL_FREQUENCY_WEIGHTED_FG] == pytest.approx(
        2.0 / 3.0
    )
    assert metrics["IoU_metrics"][
        IOU_GLOBAL_FREQUENCY_WEIGHTED_ALL_CLASS
    ] == pytest.approx(29.0 / 48.0)


def test_scene_id_inference_covers_current_dataset_suffixes() -> None:
    assert infer_segmentation_scene_id("subject-10_p13") == "subject-10"
    assert infer_segmentation_scene_id("case-roi3_roi_0") == "case-roi3"
    assert infer_segmentation_scene_id("tma-case_p1_0") == "tma-case"
    assert infer_segmentation_scene_id("already-a-scene") == "already-a-scene"


def test_absent_foreground_is_nan_for_new_global_protocol() -> None:
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    accumulator = SegmentationMetricAccumulator(2, [DICE_GLOBAL_FG])
    accumulator.update(target, target, scene_ids=["empty"])
    assert math.isnan(accumulator.compute()["Dice"][DICE_GLOBAL_FG])
