"""Validate that a data preflight matches the exact proposed training inputs."""
from __future__ import annotations

import json
from pathlib import Path

from .dataset_fingerprint import fingerprint_windows
from .sequence_manifest import build_sequence_manifest, split_windows


def verify_preflight_report(path, *, observations, schema, sequence_length, stride,
                            steering_limit_deg=40.0, speed_limit_kmh=120.0):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("readiness", {}).get("passed") is not True:
        raise ValueError("DBNet preflight did not pass")
    expected_schema = {
        "steering_channel": schema.steering_channel,
        "speed_channel": schema.speed_channel,
        "steering_unit": schema.steering_unit,
        "speed_unit": schema.speed_unit,
    }
    if report.get("target_schema") != expected_schema:
        raise ValueError("Preflight target schema differs from training")
    if report.get("sequence_length") != sequence_length or report.get("stride") != stride:
        raise ValueError("Preflight temporal window settings differ from training")
    expected_limits = {
        "steering_abs_deg": steering_limit_deg,
        "speed_min_kmh": 0,
        "speed_max_kmh": speed_limit_kmh,
    }
    if report.get("plausibility_limits") != expected_limits:
        raise ValueError("Preflight plausibility bounds differ from training")
    windows = split_windows(build_sequence_manifest(
        observations, sequence_length=sequence_length, stride=stride
    ))
    for split in ("train", "val"):
        group = windows[split]
        if not group:
            raise ValueError(f"Missing {split} windows")
        if report.get("window_counts", {}).get(split) != len(group):
            raise ValueError(f"{split} window count differs from preflight")
        if report.get("window_membership_sha256", {}).get(split) != fingerprint_windows(group):
            raise ValueError(f"{split} window membership differs from preflight")
    return True
