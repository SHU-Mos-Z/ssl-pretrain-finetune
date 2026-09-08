"""预训练 loss 曲线记录与 matplotlib 绘图。"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (summary 字段名, 显示标题, 单图文件名)
LOSS_SPECS: list[tuple[str, str, str]] = [
    ("loss", "Total Loss", "loss_total.png"),
    ("loss_od", "OD Reconstruction", "loss_od.png"),
    ("loss_i", "Intensity Reconstruction", "loss_i.png"),
    ("loss_cons_pix", "Pixel Consistency", "loss_cons_pix.png"),
    ("loss_cons_token", "Token Consistency (DINO)", "loss_cons_token.png"),
    ("loss_anchor", "Teacher Anchor (NMF align)", "loss_anchor.png"),
]


class PretrainLossPlotter:
    """按 epoch 累积 loss，并输出单图 + subplot 大图到 save_dir。"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.plot_dir = os.path.join(save_dir, "loss_plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.epochs: list[int] = []
        self.history: dict[str, list[float]] = {key: [] for key, _, _ in LOSS_SPECS}

    def record(self, epoch: int, summary: dict[str, Any]) -> None:
        self.epochs.append(epoch)
        for key, _, _ in LOSS_SPECS:
            self.history[key].append(float(summary[key]))

    def save(self) -> None:
        if not self.epochs:
            return

        self._save_history_json()
        for key, title, filename in LOSS_SPECS:
            self._plot_single(key, title, os.path.join(self.plot_dir, filename))
        self._plot_combined(os.path.join(self.plot_dir, "loss_curves_all.png"))

    def _save_history_json(self) -> None:
        payload = {"epochs": self.epochs, **self.history}
        path = os.path.join(self.plot_dir, "loss_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _plot_single(self, key: str, title: str, path: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.epochs, self.history[key], marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_combined(self, path: str) -> None:
        n = len(LOSS_SPECS)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
        axes_flat = axes.flatten() if n > 1 else [axes]

        for ax, (key, title, _) in zip(axes_flat, LOSS_SPECS):
            ax.plot(self.epochs, self.history[key], marker="o", markersize=3, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.suptitle("Pretrain Loss Curves", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
