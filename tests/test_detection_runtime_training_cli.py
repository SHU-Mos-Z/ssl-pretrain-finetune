from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from utils.preprocessing.offline_nmf import cache_dir_name


def test_runtime_window_training_cli_with_gradient_accumulation(tmp_path):
    root = tmp_path / "data"
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
    save_dir = tmp_path / "record"
    command = [
        sys.executable,
        "train_finetune_conditioned_detection.py",
        "--train-root", str(root), "--train-annotation", "annotations",
        "--val-root", str(root), "--val-annotation", "annotations",
        "--test-root", str(root), "--test-annotation", "annotations",
        "--detection-mode", "anchor_free", "--det-feature-mode", "gated_pyramid",
        "--detection-view-mode", "runtime_window",
        "--source-crop-size", "80", "--model-input-size", "64", "--eval-stride", "32",
        "--train-views-per-source", "2", "--positive-guided-fraction", "1",
        "--runtime-visible-ratio", "0.7", "--runtime-min-visible-side", "4",
        "--enable-crop-truncated-positive", "--eval-ownership-filter",
        "--epochs", "1", "--batch-size", "1", "--gradient-accumulation-steps", "2",
        "--workers", "0", "--no-augment", "--progress", "none", "--save-interval", "0",
        "--patch-size", "16", "--spectral-patch-size", "2",
        "--embed-dim", "16", "--vit-depth", "1", "--vit-heads", "4",
        "--mlp-ratio", "2", "--dropout", "0", "--cnn-stem-ch", "8",
        "--cnn-spectral-agg", "mean", "--fusion-heads", "4", "--feature-dim", "8",
        "--decoder-mid-ch", "8", "--residual-hidden-dim", "8",
        "--det-feature-dim", "8", "--head-depth", "1",
        "--fcos-regression-ranges", "0:8,8:16,16:32,32:100000000",
        "--pre-nms-topk", "10", "--max-detections", "5",
        "--nmf-k", "3", "--nmf-simplex", "--allow-index-wavelengths",
        "--save-dir", str(save_dir),
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, timeout=240)
    assert (save_dir / "ckpt_best.pth").is_file()
    assert (save_dir / "validation_latest" / "metrics.json").is_file()
    assert (save_dir / "test_best" / "metrics.json").is_file()
    resolved = json.loads((save_dir / "config_resolved.json").read_text(encoding="utf-8"))
    assert resolved["view_config"]["view_mode"] == "runtime_window"
    assert resolved["args"]["gradient_accumulation_steps"] == 2
