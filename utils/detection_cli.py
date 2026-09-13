"""Argument and configuration helpers for conditioned detection entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.conditioned_contracts import ConditionedModelConfig
from models.detection_contracts import DetectionConfig
from utils.datasets.detection_view_geometry import DetectionViewConfig


def _float_tuple(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Parse float sequences from CLI text or a decoded JSON array.

    ``argparse`` supplies comma-separated strings, while anchor fitting writes
    standards-compliant JSON arrays.  Keeping both representations supported
    lets one validation path serve command-line arguments and persisted anchor
    configurations.
    """
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated string or a JSON array"
        )
    try:
        values = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return values


def _range_tuple(text: str) -> tuple[tuple[float, float], ...]:
    try:
        ranges = tuple(
            tuple(float(value.strip()) for value in item.split(":"))
            for item in text.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not ranges or any(len(pair) != 2 for pair in ranges):
        raise argparse.ArgumentTypeError("ranges must look like 0:32,32:64,...")
    return ranges


def _int_pair(text: str) -> tuple[int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if len(values) == 1:
        values = (values[0], values[0])
    if len(values) != 2 or min(values) < 1:
        raise argparse.ArgumentTypeError("size must be N or height,width with positive integers")
    return values


def add_conditioned_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--spectral-patch-size", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-stem-ch", type=int, default=64)
    parser.add_argument("--cnn-spectral-agg", choices=("mean", "max", "attention"), default="attention")
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--decoder-mid-ch", type=int, default=64)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--confidence-temperature", type=float, default=0.05)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-extra", type=float, default=1.0)
    parser.add_argument("--od-max", type=float, default=3.0)


def add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detection-mode", choices=("anchor_based", "anchor_free"), default="anchor_based")
    parser.add_argument(
        "--det-feature-mode",
        choices=("z_pyramid", "gated_pyramid", "gated_fpn", "z_full"),
        default="z_pyramid",
    )
    parser.add_argument("--det-feature-dim", type=int, default=128)
    parser.add_argument("--head-depth", type=int, default=4)
    parser.add_argument("--det-head-norm", choices=("none", "group_norm"), default="none")
    parser.add_argument("--det-head-norm-groups", type=int, default=32)
    parser.add_argument("--det-quality-mode", choices=("legacy", "iou"), default="legacy")
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--quality-score-power", type=float, default=0.5)
    parser.add_argument("--anchor-sizes", type=_float_tuple, default=(16.0, 32.0, 64.0, 128.0))
    parser.add_argument("--anchor-scales", type=_float_tuple, default=(1.0, 1.2599, 1.5874))
    parser.add_argument("--anchor-ratios", type=_float_tuple, default=(0.5, 1.0, 2.0))
    parser.add_argument(
        "--anchor-config-json",
        help="Optional JSON from fit_wbc_detection_anchors.py; overrides sizes/scales/ratios.",
    )
    parser.add_argument("--anchor-offset", type=float, default=0.5)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-iou-threshold", type=float, default=0.4)
    parser.add_argument("--ignore-iou-threshold", type=float, default=0.5)
    parser.add_argument("--matcher-chunk-size", type=int, default=65536)
    parser.add_argument("--box-coder-weights", type=_float_tuple, default=(1.0, 1.0, 1.0, 1.0))
    parser.add_argument("--box-loss", choices=("smooth_l1", "giou"), default="smooth_l1")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0 / 9.0)
    parser.add_argument(
        "--fcos-regression-ranges",
        type=_range_tuple,
        default=((0.0, 32.0), (32.0, 64.0), (64.0, 128.0), (128.0, 1e8)),
    )
    parser.add_argument("--fcos-center-sampling-radius", type=float, default=1.5)
    parser.add_argument(
        "--fcos-normalize-reg-targets-by-stride",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--centerness-loss-weight", type=float, default=1.0)
    parser.add_argument("--prior-probability", type=float, default=0.01)
    parser.add_argument(
        "--score-threshold",
        "--ap-score-threshold",
        dest="score_threshold",
        type=float,
        default=0.05,
        help="Low candidate floor used for COCO AP evaluation (legacy alias retained).",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--min-box-size", type=float, default=1.0)


def add_nmf_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wavelength-file")
    parser.add_argument("--allow-index-wavelengths", action="store_true")
    parser.add_argument("--nmf-k", type=int, default=16)
    parser.add_argument("--nmf-l1", type=float, default=5e-4)
    parser.add_argument("--nmf-l2", type=float, default=2e-4)
    parser.add_argument("--nmf-l3", type=float, default=1e-2)
    parser.add_argument("--nmf-simplex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nmf-lam-e", type=float, default=0.05)
    parser.add_argument("--nmf-e-clamp-max", type=float, default=3.0)


def add_detection_view_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for reversible source-scene runtime windows.

    Defaults are resolved by :func:`detection_view_config_from_args`.  Keeping
    argparse defaults at ``None`` lets standalone evaluation inherit the exact
    contract stored in a training checkpoint.
    """

    parser.add_argument(
        "--detection-view-mode", choices=("direct", "runtime_window"), default=None
    )
    parser.add_argument("--source-crop-size", type=_int_pair, default=None)
    parser.add_argument("--model-input-size", type=_int_pair, default=None)
    parser.add_argument("--train-views-per-source", type=int, default=None)
    parser.add_argument("--positive-guided-fraction", type=float, default=None)
    parser.add_argument("--eval-stride", type=_int_pair, default=None)
    parser.add_argument("--runtime-visible-ratio", type=float, default=None)
    parser.add_argument("--runtime-min-visible-side", type=float, default=None)
    parser.add_argument(
        "--enable-crop-truncated-positive",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--eval-ownership-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--global-nms-threshold", type=float, default=None)


def detection_view_config_from_args(
    args: argparse.Namespace,
    base: DetectionViewConfig | dict | None = None,
) -> DetectionViewConfig:
    if base is None:
        values = DetectionViewConfig(seed=int(getattr(args, "seed", 42))).to_dict()
    elif isinstance(base, DetectionViewConfig):
        values = base.to_dict()
    else:
        values = DetectionViewConfig.from_dict(dict(base)).to_dict()
    mapping = {
        "detection_view_mode": "view_mode",
        "source_crop_size": "source_crop_size",
        "model_input_size": "model_input_size",
        "train_views_per_source": "train_views_per_source",
        "positive_guided_fraction": "positive_guided_fraction",
        "eval_stride": "eval_stride",
        "runtime_visible_ratio": "visible_ratio_threshold",
        "runtime_min_visible_side": "min_visible_side",
        "enable_crop_truncated_positive": "enable_crop_truncated_positive",
        "eval_ownership_filter": "ownership_filter",
        "global_nms_threshold": "global_nms_threshold",
    }
    for argument_name, config_name in mapping.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            values[config_name] = value
    values["seed"] = int(getattr(args, "seed", values.get("seed", 42)))
    config = DetectionViewConfig.from_dict(values)
    return config


def model_config_from_args(args: argparse.Namespace) -> ConditionedModelConfig:
    return ConditionedModelConfig(
        patch_size=args.patch_size,
        spectral_patch_size=args.spectral_patch_size,
        embed_dim=args.embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        cnn_stem_ch=args.cnn_stem_ch,
        cnn_spectral_agg=args.cnn_spectral_agg,
        fusion_heads=args.fusion_heads,
        feature_dim=args.feature_dim,
        decoder_mid_ch=args.decoder_mid_ch,
        residual_hidden_dim=args.residual_hidden_dim,
        ridge_lambda=args.ridge_lambda,
        confidence_temperature=args.confidence_temperature,
        alpha_min=args.alpha_min,
        alpha_extra=args.alpha_extra,
        od_max=args.od_max,
    )


def detection_config_from_args(args: argparse.Namespace, num_classes: int) -> DetectionConfig:
    anchor_sizes = tuple(args.anchor_sizes)
    anchor_scales = tuple(args.anchor_scales)
    anchor_ratios = tuple(args.anchor_ratios)
    if getattr(args, "anchor_config_json", None):
        path = Path(args.anchor_config_json).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        suggested = payload.get("suggested_anchor_config", payload)
        try:
            anchor_sizes = _float_tuple(suggested["anchor_sizes"])
            anchor_scales = _float_tuple(suggested["anchor_scales"])
            anchor_ratios = _float_tuple(suggested["anchor_ratios"])
        except (KeyError, TypeError, ValueError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"invalid anchor config JSON: {path}") from error
    config = DetectionConfig(
        detection_mode=args.detection_mode,
        feature_mode=args.det_feature_mode,
        num_classes=num_classes,
        det_feature_dim=args.det_feature_dim,
        head_depth=args.head_depth,
        head_norm=args.det_head_norm,
        head_norm_groups=args.det_head_norm_groups,
        quality_mode=args.det_quality_mode,
        quality_loss_weight=args.quality_loss_weight,
        quality_score_power=args.quality_score_power,
        anchor_sizes=anchor_sizes,
        anchor_scales=anchor_scales,
        anchor_ratios=anchor_ratios,
        anchor_offset=args.anchor_offset,
        positive_iou_threshold=args.positive_iou_threshold,
        negative_iou_threshold=args.negative_iou_threshold,
        ignore_iou_threshold=args.ignore_iou_threshold,
        matcher_chunk_size=args.matcher_chunk_size,
        box_coder_weights=tuple(args.box_coder_weights),
        box_loss=args.box_loss,
        smooth_l1_beta=args.smooth_l1_beta,
        fcos_regression_ranges=tuple(tuple(pair) for pair in args.fcos_regression_ranges),
        fcos_center_sampling_radius=args.fcos_center_sampling_radius,
        fcos_normalize_reg_targets_by_stride=args.fcos_normalize_reg_targets_by_stride,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        box_loss_weight=args.box_loss_weight,
        centerness_loss_weight=args.centerness_loss_weight,
        prior_probability=args.prior_probability,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        pre_nms_topk=args.pre_nms_topk,
        max_detections=args.max_detections,
        min_box_size=args.min_box_size,
    )
    config.validate()
    return config


def dataset_kwargs_from_args(
    args: argparse.Namespace, view_config: DetectionViewConfig | None = None
) -> dict:
    return {
        "patch_size": args.patch_size,
        "spectral_patch_size": args.spectral_patch_size,
        "nmf_k": args.nmf_k,
        "nmf_l1": args.nmf_l1,
        "nmf_l2": args.nmf_l2,
        "nmf_l3": args.nmf_l3,
        "nmf_simplex": args.nmf_simplex,
        "nmf_lam_e": args.nmf_lam_e,
        "nmf_e_clamp_max": args.nmf_e_clamp_max,
        "wavelength_file": args.wavelength_file,
        "allow_index_wavelengths": args.allow_index_wavelengths,
        "od_max": args.od_max,
        "seed": getattr(args, "seed", 42),
        "view_config": view_config or detection_view_config_from_args(args),
    }
