"""Shared parser for exact classification-sample exclusion lists."""

from __future__ import annotations

import json
from pathlib import Path


def load_excluded_samples(path: str | Path) -> set[tuple[str, str]]:
    """Load unique ``(class_name, stem)`` identities from a JSON record list.

    Each record must contain non-empty string fields named ``class_name`` and
    ``stem``. Additional audit fields are accepted and ignored.
    """
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"sample-exclusion JSON does not exist: {json_path}")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("sample-exclusion JSON must be a top-level record list")

    identities: set[tuple[str, str]] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise TypeError(
                f"sample-exclusion record {index} must be an object, got "
                f"{type(record).__name__}"
            )
        class_name = record.get("class_name")
        stem = record.get("stem")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                f"sample-exclusion record {index} has no valid 'class_name'"
            )
        if not isinstance(stem, str) or not stem:
            raise ValueError(f"sample-exclusion record {index} has no valid 'stem'")
        identity = (class_name, stem)
        if identity in identities:
            raise ValueError(
                "sample-exclusion JSON contains a duplicate identity: "
                f"class_name={class_name!r}, stem={stem!r}"
            )
        identities.add(identity)
    return identities
