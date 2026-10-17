"""Split-safe temporal sequence manifest for synchronized driving observations.

Generic implementation schema, NOT a claim about DBNet's original file layout.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    sample_id: str
    drive_id: str
    frame_index: int
    split: str
    camera_path: str
    lidar_path: str
    steering_deg: float
    speed_kmh: float


def validate_observations(observations: Iterable[Observation]) -> list[Observation]:
    import math
    records = list(observations)
    if not records:
        raise ValueError("Observation manifest is empty")
    seen = set()
    drive_splits: dict[str, str] = {}
    for row in records:
        if row.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {row.split}")
        if not all((row.sample_id, row.drive_id, row.camera_path, row.lidar_path)):
            raise ValueError("Missing sample, drive, camera, or LiDAR identifier")
        if row.frame_index < 0 or not math.isfinite(row.steering_deg) or not math.isfinite(row.speed_kmh):
            raise ValueError("Invalid frame index or target")
        if row.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {row.sample_id}")
        seen.add(row.sample_id)
        previous = drive_splits.setdefault(row.drive_id, row.split)
        if previous != row.split:
            raise ValueError(f"Drive {row.drive_id} appears in multiple splits")
    return records


def build_sequence_manifest(observations: Iterable[Observation], *,
                            sequence_length: int, stride: int = 1
                            ) -> list[tuple[Observation, ...]]:
    """Construct windows within a drive/split; targets belong to last frame.

    Last-frame target alignment and stride are implementation choices.
    Missing frame indices break a window rather than silently skipping gaps.
    """
    if sequence_length < 2 or stride < 1:
        raise ValueError("sequence_length must be >=2 and stride >=1")
    rows = validate_observations(observations)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.split, row.drive_id)].append(row)
    windows = []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r.frame_index)
        indices = [r.frame_index for r in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate frame_index within a drive")
        for start in range(0, len(ordered) - sequence_length + 1, stride):
            window = ordered[start:start + sequence_length]
            if all(b.frame_index == a.frame_index + 1 for a, b in zip(window, window[1:])):
                windows.append(tuple(window))
    return windows


def split_windows(windows: Iterable[tuple[Observation, ...]]) -> dict[str, list[tuple[Observation, ...]]]:
    result = {"train": [], "val": [], "test": []}
    for window in windows:
        if not window or len({r.split for r in window}) != 1:
            raise ValueError("A window must belong to exactly one split")
        result[window[0].split].append(window)
    return result
