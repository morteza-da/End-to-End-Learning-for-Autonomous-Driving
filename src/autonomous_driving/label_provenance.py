"""Explicit provenance gate for independently verified target-channel semantics."""
from __future__ import annotations

import json
from pathlib import Path


def load_verified_label_provenance(path, schema):
    """Require an independently reviewed source and matching target mapping.

    This validates completeness and consistency, not truthfulness of the source.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Label provenance must be a JSON object")
    required = ("source_title", "source_url", "evidence_excerpt",
                "reviewed_by", "review_date", "steering_channel",
                "speed_channel", "steering_unit", "speed_unit")
    if any(key not in payload for key in required):
        raise ValueError("Missing required label provenance field")
    for key in ("source_title", "source_url", "evidence_excerpt", "reviewed_by", "review_date"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Missing nonempty evidence field: {key}")
    if not payload["source_url"].startswith(("https://", "http://")):
        raise ValueError("Source URL must be HTTP(S)")
    for key in ("steering_channel", "speed_channel", "steering_unit", "speed_unit"):
        if type(payload[key]) is not type(getattr(schema, key)) or payload[key] != getattr(schema, key):
            raise ValueError(f"Provenance disagrees with configured {key}")
    return payload
