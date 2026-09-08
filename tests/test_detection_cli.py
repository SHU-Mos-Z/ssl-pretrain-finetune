from __future__ import annotations

import argparse
import json

import pytest

from utils.detection_cli import (
    _float_tuple,
    add_detection_arguments,
    detection_config_from_args,
)


def _detection_args(*extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_detection_arguments(parser)
    return parser.parse_args(list(extra))


def test_float_tuple_accepts_cli_text_and_json_sequences():
    expected = (16.0, 32.0, 64.0, 128.0)
    assert _float_tuple("16,32,64,128") == expected
    assert _float_tuple([16, 32.0, "64", 128]) == expected
    assert _float_tuple((16, 32, 64, 128)) == expected


def test_anchor_config_json_accepts_fitter_array_output(tmp_path):
    anchor_path = tmp_path / "anchors.json"
    anchor_path.write_text(
        json.dumps(
            {
                "suggested_anchor_config": {
                    "anchor_sizes": [23.713, 37.549, 55.992, 93.8],
                    "anchor_scales": [0.8, 1.0, 1.25],
                    "anchor_ratios": [0.508, 0.84, 1.316],
                }
            }
        ),
        encoding="utf-8",
    )
    args = _detection_args("--anchor-config-json", str(anchor_path))

    config = detection_config_from_args(args, num_classes=1)

    assert config.anchor_sizes == (23.713, 37.549, 55.992, 93.8)
    assert config.anchor_scales == (0.8, 1.0, 1.25)
    assert config.anchor_ratios == (0.508, 0.84, 1.316)


def test_invalid_anchor_config_json_has_contextual_error(tmp_path):
    anchor_path = tmp_path / "invalid.json"
    anchor_path.write_text(
        json.dumps(
            {
                "suggested_anchor_config": {
                    "anchor_sizes": {"not": "an array"},
                    "anchor_scales": [1.0],
                    "anchor_ratios": [1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    args = _detection_args("--anchor-config-json", str(anchor_path))

    with pytest.raises(ValueError, match="invalid anchor config JSON"):
        detection_config_from_args(args, num_classes=1)
