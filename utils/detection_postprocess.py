"""Decode dense raw predictions and perform class-aware NMS."""

from __future__ import annotations

import torch

from models.detection_contracts import DetectionConfig
from models.modules_detection import BoxCoder
from models.modules_detection.box_ops import batched_nms, clip_boxes_to_image, remove_small_boxes


class DetectionPostProcessor:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.box_coder = BoxCoder(config.box_coder_weights)

    def _finish(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        boxes = clip_boxes_to_image(boxes, image_size)
        keep = remove_small_boxes(boxes, self.config.min_box_size)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = batched_nms(boxes, scores, labels, self.config.nms_threshold)
        keep = keep[: self.config.max_detections]
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def _anchor_image(self, output: dict, index: int, image_size: tuple[int, int]):
        boxes, scores, labels = [], [], []
        for level_index, (anchors, logits, deltas) in enumerate(zip(
            output["anchors"], output["cls_logits"], output["bbox_deltas"]
        )):
            class_probabilities = logits[index].float().sigmoid()
            if self.config.quality_mode == "iou":
                quality = output["quality_logits"][level_index][index].float().sigmoid()[:, None]
                power = self.config.quality_score_power
                class_probabilities = (
                    class_probabilities.clamp(min=1e-8).pow(1.0 - power)
                    * quality.clamp(min=1e-8).pow(power)
                )
            probabilities = class_probabilities.flatten()
            keep = torch.where(probabilities >= self.config.score_threshold)[0]
            if len(keep) > self.config.pre_nms_topk:
                _, order = probabilities[keep].topk(self.config.pre_nms_topk)
                keep = keep[order]
            anchor_indices = torch.div(keep, self.config.num_classes, rounding_mode="floor")
            class_indices = keep.remainder(self.config.num_classes)
            boxes.append(self.box_coder.decode(anchors[anchor_indices], deltas[index, anchor_indices].float()))
            scores.append(probabilities[keep])
            labels.append(class_indices)
        return self._finish(
            torch.cat(boxes), torch.cat(scores), torch.cat(labels), image_size
        )

    def _fcos_image(self, output: dict, index: int, image_size: tuple[int, int]):
        boxes, scores, labels = [], [], []
        centerness_levels = output.get("centerness_logits", [None] * len(output["points"]))
        quality_levels = output.get("quality_logits", [None] * len(output["points"]))
        for points, point_strides, logits, regression, centerness, quality in zip(
            output["points"],
            output["point_strides"],
            output["cls_logits"],
            output["bbox_regression"],
            centerness_levels,
            quality_levels,
        ):
            class_probabilities = logits[index].float().sigmoid()
            auxiliary = centerness if centerness is not None else quality
            if auxiliary is None:
                raise KeyError("FCOS output lacks centerness/quality logits")
            auxiliary_probability = auxiliary[index].float().sigmoid()[:, None]
            if centerness is not None:
                probabilities = torch.sqrt(
                    class_probabilities * auxiliary_probability
                ).flatten()
            else:
                power = self.config.quality_score_power
                probabilities = (
                    class_probabilities.clamp(min=1e-8).pow(1.0 - power)
                    * auxiliary_probability.clamp(min=1e-8).pow(power)
                ).flatten()
            keep = torch.where(probabilities >= self.config.score_threshold)[0]
            if len(keep) > self.config.pre_nms_topk:
                _, order = probabilities[keep].topk(self.config.pre_nms_topk)
                keep = keep[order]
            point_indices = torch.div(keep, self.config.num_classes, rounding_mode="floor")
            class_indices = keep.remainder(self.config.num_classes)
            selected_regression = regression[index, point_indices].float()
            if self.config.fcos_normalize_reg_targets_by_stride:
                selected_regression = selected_regression * point_strides[point_indices, None]
            selected_points = points[point_indices]
            boxes.append(
                torch.stack(
                    (
                        selected_points[:, 0] - selected_regression[:, 0],
                        selected_points[:, 1] - selected_regression[:, 1],
                        selected_points[:, 0] + selected_regression[:, 2],
                        selected_points[:, 1] + selected_regression[:, 3],
                    ),
                    dim=1,
                )
            )
            scores.append(probabilities[keep])
            labels.append(class_indices)
        return self._finish(
            torch.cat(boxes), torch.cat(scores), torch.cat(labels), image_size
        )

    @torch.no_grad()
    def __call__(
        self, output: dict, image_sizes: list[tuple[int, int]] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        if output.get("detection_mode") != self.config.detection_mode:
            raise ValueError("model output and postprocessor detection modes differ")
        batch_size = output["cls_logits"][0].shape[0]
        if image_sizes is None:
            image_sizes = [tuple(output["image_size"])] * batch_size
        if len(image_sizes) != batch_size:
            raise ValueError("image_sizes does not match detector batch size")
        decode = self._anchor_image if self.config.detection_mode == "anchor_based" else self._fcos_image
        return [decode(output, index, tuple(image_sizes[index])) for index in range(batch_size)]
