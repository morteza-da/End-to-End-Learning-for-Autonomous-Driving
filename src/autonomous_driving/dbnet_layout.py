"""Inspect original DBNet-2018 extracted train/val/test folders.

Original README: drive folders contain dvr_66x200/, points_16384/,
and behavior.csv (labels not supplied for test in README).
No raw dataset redistribution is performed.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DBNetDriveAudit:
    split: str
    drive: str
    camera_count: int
    lidar_count: int
    label_rows: int | None
    paired_count: int
    missing_camera: tuple[int, ...]
    missing_lidar: tuple[int, ...]
    missing_labels: tuple[int, ...]


def numbered_files(folder: Path, extension: str) -> dict[int, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing DBNet folder: {folder}")
    result = {}
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == extension:
            if not file.stem.isdecimal():
                raise ValueError(f"Non-numeric DBNet frame name: {file}")
            index = int(file.stem)
            if index in result:
                raise ValueError(f"Duplicate DBNet frame index: {index}")
            result[index] = file
    return result


def read_behavior_rows(path: Path) -> list[tuple[str, ...]]:
    """Read positional CSV values without assuming header or units."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [tuple(value.strip() for value in row) for row in csv.reader(stream)
                if row and any(value.strip() for value in row)]


def audit_drive(drive_path: Path, split: str) -> DBNetDriveAudit:
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    cameras = numbered_files(drive_path / "dvr_66x200", ".jpg")
    clouds = numbered_files(drive_path / "points_16384", ".las")
    behavior_path = drive_path / "behavior.csv"
    labels = read_behavior_rows(behavior_path) if behavior_path.exists() else None
    if labels is not None:
        from .label_inspection import inspect_behavior_csv
        label_report = inspect_behavior_csv(behavior_path, expected_frames=len(cameras | clouds))
        if label_report['orientation'] == 'frame_count_mismatch':
            raise ValueError(f'Label/frame count mismatch: {behavior_path}')
    if labels is None and split != "test":
        raise FileNotFoundError(f"Missing behavior.csv for labeled split: {drive_path}")
    camera_ids, lidar_ids = set(cameras), set(clouds)
    paired = camera_ids & lidar_ids
    expected = set(range(max(camera_ids | lidar_ids, default=-1) + 1))
    label_frame_count = (None if labels is None else
                         len(labels) if label_report['orientation'] == 'frames_as_rows'
                         else label_report['shape'][1])
    missing_labels = tuple(sorted(i for i in paired if label_frame_count is not None and i >= label_frame_count))
    return DBNetDriveAudit(
        split, drive_path.name, len(cameras), len(clouds),
        label_frame_count, len(paired),
        tuple(sorted((lidar_ids | expected) - camera_ids)),
        tuple(sorted((camera_ids | expected) - lidar_ids)),
        missing_labels,
    )


def audit_dataset(root: Path) -> list[DBNetDriveAudit]:
    """Scan extracted dataset. No need to copy images or LAS into GitHub."""
    root = Path(root)
    audits = []
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing DBNet split: {folder}")
        for drive in sorted((x for x in folder.iterdir() if x.is_dir()),
                            key=lambda p: (not p.name.isdecimal(),
                                           int(p.name) if p.name.isdecimal() else p.name)):
            audits.append(audit_drive(drive, split))
    return audits


def build_labeled_observations(root: Path):
    """Map original positional labels to generic records.

    Column 0/1 units and semantic order MUST be independently verified before
    using steering_deg/speed_kmh; returns raw label strings instead.
    """
    root = Path(root)
    rows = []
    for split in ("train", "val"):
        for drive in sorted((root / split).iterdir()):
            if not drive.is_dir():
                continue
            cameras = numbered_files(drive / "dvr_66x200", ".jpg")
            lidars = numbered_files(drive / "points_16384", ".las")
            from .label_inspection import inspect_behavior_csv, extract_raw_frame_labels
            label_path = drive / "behavior.csv"
            report = inspect_behavior_csv(label_path, expected_frames=len(set(cameras) | set(lidars)))
            if report['orientation'] not in {'frames_as_rows', 'frames_as_columns'}:
                raise ValueError(f'Unrecognized label orientation: {label_path}')
            labels = extract_raw_frame_labels(
                label_path, frame_count=len(set(cameras) | set(lidars)),
                orientation=report['orientation'])
            for index in sorted(set(cameras) & set(lidars)):
                if index >= len(labels):
                    raise ValueError(f"Missing label at {split}/{drive.name}/{index}")
                rows.append({
                    "sample_id": f"{split}/{drive.name}/{index}",
                    "split": split,
                    "drive_id": f"{split}/{drive.name}",
                    "frame_index": index,
                    "camera_path": str(cameras[index].relative_to(root)),
                    "lidar_path": str(lidars[index].relative_to(root)),
                    "raw_label_columns": labels[index],
                })
    return rows
