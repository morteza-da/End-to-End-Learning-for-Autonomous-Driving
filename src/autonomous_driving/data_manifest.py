"""Validate explicit sample manifests without loading private research data.

Schema (CSV): sample_id,split,camera_path,lidar_path,steering,speed
Paths are relative to a dataset root; this module does not read sensor contents.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

REQUIRED = ("sample_id", "split", "camera_path", "lidar_path", "steering", "speed")
SPLITS = {"train", "val", "test"}


def validate_manifest(path: str | Path, dataset_root: str | Path | None = None) -> dict[str, int]:
    """Validate schema, labels, split names, IDs, and optionally file existence.

    Raises ValueError with a row-specific explanation. Does not assume units,
    sensor calibration, or dataset-specific acceptable label ranges.
    """
    counts = {name: 0 for name in sorted(SPLITS)}
    seen: set[str] = set()
    manifest = Path(path)
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in REQUIRED):
            raise ValueError(f"Manifest requires columns: {', '.join(REQUIRED)}")
        for row_num, row in enumerate(reader, start=2):
            sample_id = (row["sample_id"] or "").strip()
            split = (row["split"] or "").strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"Row {row_num}: missing or duplicate sample_id")
            seen.add(sample_id)
            if split not in SPLITS:
                raise ValueError(f"Row {row_num}: invalid split {split!r}")
            for field in ("steering", "speed"):
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Row {row_num}: invalid {field}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Row {row_num}: non-finite {field}")
            for field in ("camera_path", "lidar_path"):
                raw = (row[field] or "").strip()
                if not raw:
                    raise ValueError(f"Row {row_num}: missing {field}")
                relative = Path(raw)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Row {row_num}: unsafe {field} path")
                if root is not None:
                    resolved = (root / relative).resolve()
                    if not resolved.is_relative_to(root) or not resolved.is_file():
                        raise ValueError(f"Row {row_num}: missing or unsafe file for {field}")
            counts[split] += 1
    if not seen:
        raise ValueError("Manifest is empty")
    return counts
