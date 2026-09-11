"""Runtime helpers shared by detection training and standalone evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from utils.detection_metrics import evaluate_coco_detections
from utils.detection_metrics import detections_to_coco
from utils.detection_postprocess import DetectionPostProcessor
from utils.detection_visualization import render_detection_examples
from models.modules_detection.box_ops import box_iou, sample_mask_at_points
from models.modules_detection.box_ops import batched_nms, clip_boxes_to_image


def _detections_in_source_coordinates(
    detection: dict[str, torch.Tensor], target: dict[str, Any], *, ownership_filter: bool
) -> dict[str, torch.Tensor]:
    """Undo crop/resize geometry and optionally retain one tile's ownership cell."""

    boxes = detection["boxes"].float().clone()
    if len(boxes):
        scale_xy = target["scale_xy"].to(boxes.device).float()
        crop = target["crop_xyxy"].to(boxes.device).float()
        boxes[:, 0::2] = boxes[:, 0::2] / scale_xy[0] + crop[0]
        boxes[:, 1::2] = boxes[:, 1::2] / scale_xy[1] + crop[1]
        source_size = tuple(int(value) for value in target["source_size"].tolist())
        boxes = clip_boxes_to_image(boxes, source_size)
        keep = torch.ones((len(boxes),), dtype=torch.bool, device=boxes.device)
        if ownership_filter:
            ownership = target["ownership_xyxy"].to(boxes.device).float()
            centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            keep &= (
                (centers[:, 0] >= ownership[0])
                & (centers[:, 0] < ownership[2])
                & (centers[:, 1] >= ownership[1])
                & (centers[:, 1] < ownership[3])
            )
        boxes = boxes[keep]
    else:
        keep = torch.empty((0,), dtype=torch.bool, device=boxes.device)
    return {
        "boxes": boxes,
        "scores": detection["scores"][keep],
        "labels": detection["labels"][keep],
    }


def merge_source_detections(
    detections: list[dict[str, Any]], *, nms_threshold: float, max_detections: int
) -> list[dict[str, Any]]:
    """Class-aware global NMS after all runtime views have reached source space."""

    grouped: dict[int, dict[str, Any]] = {}
    for item in detections:
        image_id = int(item["image_id"])
        destination = grouped.setdefault(
            image_id,
            {"image_id": image_id, "boxes": [], "scores": [], "labels": []},
        )
        destination["boxes"].extend(item["boxes"])
        destination["scores"].extend(item["scores"])
        destination["labels"].extend(item["labels"])
    output: list[dict[str, Any]] = []
    for item in grouped.values():
        boxes = torch.as_tensor(item["boxes"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(item["scores"], dtype=torch.float32).reshape(-1)
        labels = torch.as_tensor(item["labels"], dtype=torch.int64).reshape(-1)
        keep = batched_nms(boxes, scores, labels, nms_threshold)[:max_detections]
        output.append(
            {
                "image_id": int(item["image_id"]),
                "boxes": boxes[keep].tolist(),
                "scores": scores[keep].tolist(),
                "labels": labels[keep].tolist(),
            }
        )
    return output


def move_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def distributed_context() -> tuple[bool, int, int, int]:
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return distributed, rank, world_size, local_rank


def _communication_device(device: torch.device) -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return device
    return torch.device("cpu")


def gather_detection_payload(
    local: list[dict[str, Any]], distributed: bool, rank: int, device: torch.device
):
    if not distributed:
        return local
    # torch 2.0/NCCL all_gather_object is unreliable when one eval rank has no
    # samples. Encode variable-length predictions as padded numeric tensors.
    rows: list[list[float]] = []
    for item in local:
        image_id = int(item["image_id"])
        rows.append([float(image_id), 0, 0, 0, 0, 0, -1])  # image header
        for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
            rows.append([float(image_id), *[float(v) for v in box], float(score), float(label)])
    communication_device = _communication_device(device)
    local_tensor = torch.tensor(rows, dtype=torch.float64, device=communication_device)
    if not rows:
        local_tensor = torch.empty((0, 7), dtype=torch.float64, device=communication_device)
    local_size = torch.tensor([len(local_tensor)], dtype=torch.long, device=communication_device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    max_size = max(int(value.item()) for value in sizes)
    padded = torch.zeros((max_size, 7), dtype=torch.float64, device=communication_device)
    if len(local_tensor):
        padded[: len(local_tensor)] = local_tensor
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    if rank != 0:
        return None
    output: dict[int, dict[str, Any]] = {}
    for tensor, size in zip(gathered, sizes):
        for row in tensor[: int(size.item())].cpu().tolist():
            image_id = int(row[0])
            item = output.setdefault(
                image_id,
                {"image_id": image_id, "boxes": [], "scores": [], "labels": []},
            )
            if int(row[6]) >= 0:
                item["boxes"].append(row[1:5])
                item["scores"].append(row[5])
                item["labels"].append(int(row[6]))
    return list(output.values())


def _broadcast_json(payload: dict[str, Any] | None, device: torch.device) -> dict[str, Any]:
    communication_device = _communication_device(device)
    encoded = json.dumps(payload).encode("utf-8") if dist.get_rank() == 0 else b""
    length = torch.tensor([len(encoded)], dtype=torch.long, device=communication_device)
    dist.broadcast(length, src=0)
    if dist.get_rank() == 0:
        data = torch.tensor(list(encoded), dtype=torch.uint8, device=communication_device)
    else:
        data = torch.empty((int(length.item()),), dtype=torch.uint8, device=communication_device)
    dist.broadcast(data, src=0)
    return json.loads(bytes(data.cpu().tolist()).decode("utf-8"))


@torch.no_grad()
def evaluate_detection_model(
    model: nn.Module,
    loader,
    postprocessor: DetectionPostProcessor,
    device: torch.device,
    *,
    amp: bool,
    distributed: bool,
    rank: int,
    output_dir: str | Path | None = None,
    visualization_samples: int = 0,
    pr_curve_path: str | Path | None = None,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    # The no-duplicate eval sampler may assign zero batches to a rank. Calling
    # DDP.forward() on only a subset of ranks would desynchronize its buffer
    # collectives, so evaluation uses the already-synchronized underlying model.
    evaluation_model = model.module if isinstance(model, DistributedDataParallel) else model
    dataset = loader.dataset
    view_config = getattr(dataset, "view_config", None)
    runtime_window = bool(view_config is not None and view_config.view_mode == "runtime_window")
    local: list[dict[str, Any]] = []
    for batch_index, (model_inputs, targets) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        model_inputs = move_model_inputs(model_inputs, device)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
        ):
            raw = evaluation_model(model_inputs)
        image_sizes = [tuple(int(v) for v in target["image_size"].tolist()) for target in targets]
        detections = postprocessor(raw, image_sizes)
        for detection, target in zip(detections, targets):
            # Exact custom-ignore masks suppress detections whose centers fall in
            # uncertain pixels, except reliable matches to ordinary GT.
            if len(detection["boxes"]):
                centers = 0.5 * (detection["boxes"][:, :2] + detection["boxes"][:, 2:])
                ignored = sample_mask_at_points(target["ignore_mask"].to(device), centers)
                ordinary_boxes = target["boxes"].to(device)
                ordinary_labels = target["labels"].to(device)
                protected = torch.zeros_like(ignored)
                if len(ordinary_boxes):
                    overlaps = box_iou(detection["boxes"], ordinary_boxes)
                    same_class = detection["labels"][:, None] == ordinary_labels[None, :]
                    protected = (overlaps.masked_fill(~same_class, 0).max(dim=1).values >= 0.5)
                keep = ~ignored | protected
                detection = {key: value[keep] for key, value in detection.items()}
            detection = _detections_in_source_coordinates(
                detection,
                target,
                ownership_filter=bool(runtime_window and view_config.ownership_filter),
            )
            local.append(
                {
                    "image_id": int(target["source_image_id"].item()),
                    "boxes": detection["boxes"].float().cpu().tolist(),
                    "scores": detection["scores"].float().cpu().tolist(),
                    "labels": detection["labels"].long().cpu().tolist(),
                }
            )
    gathered = gather_detection_payload(local, distributed, rank, device)
    result: list[dict[str, Any] | None] = [None]
    if rank == 0:
        assert gathered is not None
        if runtime_window:
            gathered = merge_source_detections(
                gathered,
                nms_threshold=float(view_config.global_nms_threshold),
                max_detections=int(postprocessor.config.max_detections),
            )
        if max_batches > 0:
            # Smoke tests intentionally traverse only a prefix of each rank's
            # evaluation shard.  The strict COCO adapter still requires one
            # prediction record per source image, so represent unvisited
            # sources as empty predictions.  Full evaluation (max_batches=0)
            # never enters this debug-only path.
            observed_ids = {int(item["image_id"]) for item in gathered}
            gathered.extend(
                {
                    "image_id": int(image_id),
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                }
                for image_id in dataset.image_ids
                if int(image_id) not in observed_ids
            )
        gathered.sort(key=lambda item: int(item["image_id"]))
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
        result[0] = evaluate_coco_detections(
            dataset.source_coco,
            gathered,
            dataset.label_to_category_id,
            output_json=(output_path / "predictions_coco.json") if output_path else None,
            pr_curve_path=pr_curve_path,
        )
        if output_path is not None and visualization_samples > 0:
            prediction_records = detections_to_coco(
                gathered, dataset.label_to_category_id
            )
            render_detection_examples(
                dataset.root,
                dataset.source_coco,
                prediction_records,
                output_path / "visualizations",
                visualization_samples,
            )
        if output_path is not None and runtime_window:
            (output_path / "evaluation_view_manifest.json").write_text(
                json.dumps(dataset.evaluation_view_manifest, indent=2), encoding="utf-8"
            )
        if output_path is not None:
            (output_path / "metrics.json").write_text(
                json.dumps(result[0], indent=2), encoding="utf-8"
            )
    if distributed:
        result[0] = _broadcast_json(result[0], device)
    assert result[0] is not None
    return result[0]
