"""Audit camera/LiDAR pairing by numeric frame IDs without reading media.

Matching filenames provide index-level pairing, not evidence of timestamp
synchronization or sensor calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .dbnet_layout import numbered_files


@dataclass(frozen=True)
class PairingAudit:
    split: str
    drive: str
    camera_count: int
    lidar_count: int
    paired_count: int
    camera_only_ids: tuple[int, ...]
    lidar_only_ids: tuple[int, ...]
    zero_byte_camera_ids: tuple[int, ...]
    zero_byte_lidar_ids: tuple[int, ...]
    consecutive_pair_runs: tuple[tuple[int, int], ...]

    def as_dict(self):
        return asdict(self)


def _runs(indices):
    if not indices:
        return ()
    result = []
    start = end = indices[0]
    for index in indices[1:]:
        if index == end + 1:
            end = index
        else:
            result.append((start, end))
            start = end = index
    result.append((start, end))
    return tuple(result)


def audit_pairing_drive(drive_path, *, split):
    """Summarize frame-index pairing, file sizes and contiguous paired runs."""
    drive = Path(drive_path)
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    cameras = numbered_files(drive / "dvr_66x200", ".jpg")
    lidars = numbered_files(drive / "points_16384", ".las")
    paired = sorted(cameras.keys() & lidars.keys())
    return PairingAudit(
        split=split,
        drive=drive.name,
        camera_count=len(cameras),
        lidar_count=len(lidars),
        paired_count=len(paired),
        camera_only_ids=tuple(sorted(cameras.keys() - lidars.keys())),
        lidar_only_ids=tuple(sorted(lidars.keys() - cameras.keys())),
        zero_byte_camera_ids=tuple(sorted(i for i, p in cameras.items() if p.stat().st_size == 0)),
        zero_byte_lidar_ids=tuple(sorted(i for i, p in lidars.items() if p.stat().st_size == 0)),
        consecutive_pair_runs=_runs(paired),
    )


def audit_pairing_dataset(root):
    """Audit each available DBNet split; require train/val/test directories."""
    root = Path(root)
    reports = []
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for drive in sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name):
            reports.append(audit_pairing_drive(drive, split=split))
    return reports
