"""Standalone COCO evaluation for a conditioned detection checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.conditioned_contracts import ConditionedModelConfig
from models.detection_contracts import DetectionConfig
from models.finetune_model_conditioned_detection import ConditionedDetectionModel
from utils.datasets.conditioned_detection_dataset import ConditionedDetectionDataset, DistributedEvalSampler, conditioned_detection_collate
from utils.detection_postprocess import DetectionPostProcessor
from utils.detection_runtime import distributed_context, evaluate_detection_model
from utils.detection_cli import (
    add_detection_view_arguments,
    detection_view_config_from_args,
)
from utils.detection_reporting import candidate_statistics
from torch.utils.data import DataLoader


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditioned HSI detector")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wavelength-file")
    parser.add_argument("--allow-index-wavelengths", action="store_true")
    parser.add_argument("--nmf-cache-dir")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--pre-nms-topk", type=int)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--visualization-samples", type=int, default=12)
    add_detection_view_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.is_dir():
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("checkpoint-directory evaluation must be launched without torchrun")
        checkpoints = sorted(checkpoint_path.glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"no .pth checkpoints in {checkpoint_path}")
        root_output = Path(args.output_dir)
        root_output.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in checkpoints:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint",
                str(path),
                "--data-root",
                args.data_root,
                "--annotation",
                args.annotation,
                "--output-dir",
                str(root_output / path.stem),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--visualization-samples",
                str(args.visualization_samples),
            ]
            for option, value in (
                ("--wavelength-file", args.wavelength_file),
                ("--nmf-cache-dir", args.nmf_cache_dir),
                ("--score-threshold", args.score_threshold),
                ("--nms-threshold", args.nms_threshold),
                ("--pre-nms-topk", args.pre_nms_topk),
                ("--max-detections", args.max_detections),
            ):
                if value is not None:
                    command.extend((option, str(value)))
            for option, value in (
                ("--detection-view-mode", args.detection_view_mode),
                ("--source-crop-size", args.source_crop_size),
                ("--model-input-size", args.model_input_size),
                ("--train-views-per-source", args.train_views_per_source),
                ("--positive-guided-fraction", args.positive_guided_fraction),
                ("--eval-stride", args.eval_stride),
                ("--runtime-visible-ratio", args.runtime_visible_ratio),
                ("--runtime-min-visible-side", args.runtime_min_visible_side),
                ("--global-nms-threshold", args.global_nms_threshold),
            ):
                if value is not None:
                    serialized = ",".join(map(str, value)) if isinstance(value, tuple) else str(value)
                    command.extend((option, serialized))
            for enabled, positive_option, negative_option in (
                (
                    args.enable_crop_truncated_positive,
                    "--enable-crop-truncated-positive",
                    "--no-enable-crop-truncated-positive",
                ),
                (
                    args.eval_ownership_filter,
                    "--eval-ownership-filter",
                    "--no-eval-ownership-filter",
                ),
            ):
                if enabled is not None:
                    command.append(positive_option if enabled else negative_option)
            if args.allow_index_wavelengths:
                command.append("--allow-index-wavelengths")
            if args.amp:
                command.append("--amp")
            subprocess.run(command, check=True)
            metrics = json.loads(
                (root_output / path.stem / "metrics.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "checkpoint": str(path),
                    "AP50_95": metrics["AP50_95"],
                    "AP50": metrics["AP50"],
                    "AP75": metrics["AP75"],
                    "AR_100": metrics["AR_100"],
                }
            )
        with (root_output / "checkpoint_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "detection_config" not in checkpoint:
        raise KeyError("checkpoint lacks model_config or detection_config")
    model_config = ConditionedModelConfig(**checkpoint["model_config"])
    detection_payload = dict(checkpoint["detection_config"])
    for key in ("score_threshold", "nms_threshold", "pre_nms_topk", "max_detections"):
        value = getattr(args, key)
        if value is not None:
            detection_payload[key] = value
    detection_config = DetectionConfig.from_dict(detection_payload)
    training_args = checkpoint.get("args", {})
    view_config = detection_view_config_from_args(
        args, checkpoint.get("view_config", {"view_mode": "direct"})
    )

    distributed, rank, world_size, local_rank = distributed_context()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dataset = ConditionedDetectionDataset(
        args.data_root,
        args.annotation,
        patch_size=model_config.patch_size,
        spectral_patch_size=model_config.spectral_patch_size,
        nmf_k=int(training_args.get("nmf_k", 16)),
        nmf_l1=float(training_args.get("nmf_l1", 5e-4)),
        nmf_l2=float(training_args.get("nmf_l2", 2e-4)),
        nmf_l3=float(training_args.get("nmf_l3", 1e-2)),
        nmf_simplex=bool(training_args.get("nmf_simplex", True)),
        nmf_lam_e=float(training_args.get("nmf_lam_e", 0.05)),
        nmf_e_clamp_max=float(training_args.get("nmf_e_clamp_max", 3.0)),
        nmf_cache_dir=args.nmf_cache_dir,
        wavelength_file=args.wavelength_file,
        allow_index_wavelengths=args.allow_index_wavelengths,
        od_max=model_config.od_max,
        augment=False,
        training=False,
        view_config=view_config,
    )
    expected_mapping = {
        int(key): int(value) for key, value in checkpoint["category_id_to_label"].items()
    }
    if dataset.category_id_to_label != expected_mapping:
        raise ValueError(
            f"evaluation category mapping {dataset.category_id_to_label} differs from "
            f"checkpoint {expected_mapping}"
        )
    sampler = DistributedEvalSampler(dataset, world_size, rank) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=conditioned_detection_collate,
    )
    model = ConditionedDetectionModel(model_config, detection_config).to(device)
    model.load_state_dict(checkpoint["model"])
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_config.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "data_root": str(Path(args.data_root).resolve()),
                    "annotation": args.annotation,
                    "detection_config": detection_config.to_dict(),
                    "view_config": view_config.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "candidate_statistics.json").write_text(
            json.dumps(
                candidate_statistics(model.module if isinstance(model, DDP) else model, detection_config, dataset.spatial_shape),
                indent=2,
            ),
            encoding="utf-8",
        )
    metrics = evaluate_detection_model(
        model,
        loader,
        DetectionPostProcessor(detection_config),
        device,
        amp=args.amp and device.type == "cuda",
        distributed=distributed,
        rank=rank,
        output_dir=output_dir if rank == 0 else None,
        visualization_samples=args.visualization_samples,
    )
    if rank == 0:
        print(
            f"AP50:95={metrics['AP50_95']:.4f} AP50={metrics['AP50']:.4f} "
            f"AP75={metrics['AP75']:.4f}",
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
