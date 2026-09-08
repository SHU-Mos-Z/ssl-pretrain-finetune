from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_SPECS = [
    ("loss_total", "Total Loss"),
    ("loss_od", "Masked OD Reconstruction"),
    ("loss_i", "Masked Intensity Reconstruction"),
    ("loss_c", "NMF Abundance Consistency"),
    ("loss_token", "Masked Token Reconstruction"),
    ("loss_feature", "Multi-mask Feature Consistency"),
    ("loss_delta", "Residual Magnitude Regularization"),
    ("loss_sam", "Spectral Angle"),
]


class ConditionedPretrainMonitor:
    def __init__(self, save_dir: str):
        self.path = Path(save_dir) / "loss_history.json"
        self.plot_dir = Path(save_dir) / "loss_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else []

    def record(self, epoch: int, summary: dict) -> None:
        self.rows.append({"epoch": epoch, **{k: float(v) for k, v in summary.items()}})
        self.path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        self.save_plots()

    def save_plots(self) -> None:
        if not self.rows:
            return
        epochs = [row["epoch"] for row in self.rows]
        available = [(key, title) for key, title in LOSS_SPECS if key in self.rows[0]]

        for key, title in available:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epochs, [row[key] for row in self.rows], linewidth=1.8)
            ax.set(xlabel="Epoch", ylabel="Loss", title=title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.plot_dir / f"{key}.png", dpi=150)
            plt.close(fig)

        ncols = 2
        nrows = (len(available) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
        axes = axes.reshape(-1)
        for ax, (key, title) in zip(axes, available):
            ax.plot(epochs, [row[key] for row in self.rows], linewidth=1.6)
            ax.set(xlabel="Epoch", ylabel="Loss", title=title)
            ax.grid(True, alpha=0.3)
        for ax in axes[len(available):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "loss_curves_all.png", dpi=150)
        plt.close(fig)

        focus_keys = ("loss_total", "loss_od", "loss_i")
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in focus_keys:
            ax.plot(epochs, [row[key] for row in self.rows], label=key, linewidth=1.8)
        ax.set(xlabel="Epoch", ylabel="Loss", title="Conditioned Pretraining Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_dir / "loss_total_od_i.png", dpi=150)
        plt.close(fig)
