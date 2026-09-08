"""Dataset-level metrics for single-label multi-class classification."""

from __future__ import annotations

import math

import numpy as np
import torch


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(np.bool_)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[target].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def compute_classification_metrics(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int,
) -> dict:
    logits_np = (
        logits.detach().float().cpu().numpy()
        if isinstance(logits, torch.Tensor)
        else np.asarray(logits, dtype=np.float64)
    )
    targets_np = (
        targets.detach().long().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets, dtype=np.int64)
    ).reshape(-1)
    if logits_np.ndim != 2 or logits_np.shape[1] != num_classes:
        raise ValueError("logits must have shape (N,num_classes)")
    if logits_np.shape[0] != targets_np.size:
        raise ValueError("logits and targets must contain the same number of samples")
    if targets_np.size == 0:
        raise ValueError("classification metrics require at least one sample")
    if targets_np.min() < 0 or targets_np.max() >= num_classes:
        raise ValueError("targets contain an out-of-range class index")

    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = logits_np.argmax(axis=1)
    confusion = np.bincount(
        targets_np * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    auc = np.asarray(
        [
            _binary_auc(targets_np == class_index, probabilities[:, class_index])
            for class_index in range(num_classes)
        ],
        dtype=np.float64,
    )
    valid_auc = auc[~np.isnan(auc)]
    macro_auc = float(valid_auc.mean()) if valid_auc.size else float("nan")
    return {
        "Accuracy": float((predictions == targets_np).mean()),
        "MacroF1": float(f1.mean()),
        "MacroAUC": macro_auc,
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "auc": None if math.isnan(auc[index]) else float(auc[index]),
                "support": int(actual[index]),
            }
            for index in range(num_classes)
        },
    }
