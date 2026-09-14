"""Persistent JSON/CSV histories and headless plots for fine-tuning runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PlotSpec = tuple[str, str]


class FinetuneCurveMonitor:
    """Record one flat row per epoch and refresh diagnostic curves safely."""

    def __init__(
        self,
        save_dir: str | Path,
        groups: Mapping[str, Sequence[PlotSpec]],
    ) -> None:
        self.save_dir = Path(save_dir)
        self.history_path = self.save_dir / "metrics_history.json"
        self.csv_path = self.save_dir / "metrics_history.csv"
        self.plot_dir = self.save_dir / "training_curves"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.groups = {name: tuple(specs) for name, specs in groups.items()}
        self.rows: list[dict[str, float | int]] = []
        if self.history_path.is_file():
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"fine-tuning history must be a list: {self.history_path}")
            self.rows = [dict(row) for row in payload]

    def record(self, epoch: int, values: Mapping[str, float | int]) -> None:
        row: dict[str, float | int] = {"epoch": int(epoch)}
        for key, value in values.items():
            numeric = float(value)
            row[key] = numeric if math.isfinite(numeric) else float("nan")
        self.rows = [item for item in self.rows if int(item["epoch"]) != int(epoch)]
        self.rows.append(row)
        self.rows.sort(key=lambda item: int(item["epoch"]))
        self._save_history()
        self._save_plots()

    def _save_history(self) -> None:
        self.history_path.write_text(
            json.dumps(self.rows, indent=2, allow_nan=True), encoding="utf-8"
        )
        fields = ["epoch"] + sorted(
            {key for row in self.rows for key in row if key != "epoch"}
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def _available_specs(self, specs: Sequence[PlotSpec]) -> list[PlotSpec]:
        return [
            (key, label)
            for key, label in specs
            if any(key in row for row in self.rows)
        ]

    def _draw_group(self, ax, title: str, specs: Sequence[PlotSpec]) -> bool:
        available = self._available_specs(specs)
        if not available:
            ax.axis("off")
            return False
        epochs = [int(row["epoch"]) for row in self.rows]
        for key, label in available:
            values = [float(row.get(key, float("nan"))) for row in self.rows]
            ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.6, label=label)
        ax.set(xlabel="Epoch", title=title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        return True

    def _save_plots(self) -> None:
        if not self.rows:
            return
        active_groups = [
            (name, specs)
            for name, specs in self.groups.items()
            if self._available_specs(specs)
        ]
        for name, specs in active_groups:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            self._draw_group(ax, name.replace("_", " ").title(), specs)
            fig.tight_layout()
            fig.savefig(self.plot_dir / f"{name}.png", dpi=160)
            plt.close(fig)

        if not active_groups:
            return
        ncols = 2
        nrows = (len(active_groups) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)
        flat_axes = axes.reshape(-1)
        for ax, (name, specs) in zip(flat_axes, active_groups):
            self._draw_group(ax, name.replace("_", " ").title(), specs)
        for ax in flat_axes[len(active_groups) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "training_dashboard.png", dpi=160)
        plt.close(fig)


SEGMENTATION_CURVE_GROUPS = {
    "loss_curves": (("train_loss", "Train loss"),),
    "overlap_metrics": (("val_dice", "Val Dice"), ("val_iou", "Val IoU")),
    "boundary_metric": (("val_hd95", "Val HD95"),),
    "learning_rate": (("learning_rate", "Learning rate"),),
}


def build_segmentation_curve_groups(
    dice_metric_names: Sequence[str],
    num_classes: int,
    iou_metric_names: Sequence[str] = (),
) -> dict[str, tuple[PlotSpec, ...]]:
    """Build segmentation plots for dynamic Dice and IoU protocols."""
    groups: dict[str, tuple[PlotSpec, ...]] = {
        "loss_curves": (("train_loss", "Train loss"),),
        "overlap_metrics": (
            ("val_dice_primary", "Val primary Dice"),
            ("val_iou", "Val IoU"),
        ),
        "boundary_metric": (("val_hd95", "Val HD95"),),
        "learning_rate": (("learning_rate", "Learning rate"),),
        "scene_validation_overlap": (
            ("val_scene_dice_primary", "Scene Val primary Dice"),
            ("val_scene_iou", "Scene Val IoU"),
        ),
        "scene_validation_boundary": (
            ("val_scene_hd95", "Scene Val HD95"),
        ),
    }
    scalar_specs = tuple(
        (f"val_dice_{name}", name)
        for name in dice_metric_names
        if name != "classwise"
    )
    if scalar_specs:
        groups["dice_protocols"] = scalar_specs
        groups["scene_val_dice_protocols"] = tuple(
            (f"val_scene_dice_{name}", name)
            for name in dice_metric_names
            if name != "classwise"
        )
    if "classwise" in dice_metric_names:
        groups["dice_classwise"] = tuple(
            (f"val_dice_class_{class_index}", f"class {class_index}")
            for class_index in range(num_classes)
        )
        groups["scene_val_dice_classwise"] = tuple(
            (f"val_scene_dice_class_{class_index}", f"class {class_index}")
            for class_index in range(num_classes)
        )
    iou_scalar_specs = tuple(
        (f"val_iou_{name}", name)
        for name in iou_metric_names
        if name != "classwise"
    )
    if iou_scalar_specs:
        groups["iou_protocols"] = iou_scalar_specs
        groups["scene_val_iou_protocols"] = tuple(
            (f"val_scene_iou_{name}", name)
            for name in iou_metric_names
            if name != "classwise"
        )
    if "classwise" in iou_metric_names:
        groups["iou_classwise"] = tuple(
            (f"val_iou_class_{class_index}", f"class {class_index}")
            for class_index in range(num_classes)
        )
        groups["scene_val_iou_classwise"] = tuple(
            (f"val_scene_iou_class_{class_index}", f"class {class_index}")
            for class_index in range(num_classes)
        )
    return groups

CLASSIFICATION_CURVE_GROUPS = {
    "loss_curves": (("train_loss", "Train loss"),),
    "classification_metrics": (
        ("train_accuracy", "Train accuracy"),
        ("val_accuracy", "Val accuracy"),
        ("val_macro_f1", "Val Macro-F1"),
        ("val_macro_auc", "Val Macro-AUC"),
    ),
    "learning_rate": (("learning_rate", "Learning rate"),),
}

DETECTION_CURVE_GROUPS = {
    "loss_curves": (
        ("train_loss_total", "Total loss"),
        ("train_loss_cls", "Classification loss"),
        ("train_loss_box", "Box loss"),
        ("train_loss_centerness", "Centerness loss"),
        ("train_loss_quality", "Quality loss"),
    ),
    "detection_metrics": (
        ("val_ap50_95", "Val AP50:95"),
        ("val_ap50", "Val AP50"),
        ("val_ap75", "Val AP75"),
        ("val_ar100", "Val AR100"),
    ),
    "assignment_counts": (
        ("num_positive", "Positive"),
        ("num_negative", "Negative"),
        ("num_ignored", "Ignored"),
    ),
    "learning_rate": (
        ("detector_lr", "Detector LR"),
        ("backbone_lr", "Backbone LR"),
    ),
}
