#!/usr/bin/env python3
"""Audit DBNet ZIP without extracting images or point clouds.

Usage:
  python scripts/article4_audit_zip.py /path/to/dbnet-2018.zip --output results/article4
"""
import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

PATTERN = re.compile(r"^(train|val|test)/([^/]+)/(.+)$", re.I)
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
LIDAR_EXT = {".las", ".laz"}


def read_labels(archive, member):
    if member is None:
        return None, None, "missing behavior.csv"
    try:
        rows = list(csv.reader(io.TextIOWrapper(archive.open(member), encoding="utf-8-sig")))
        return len(rows), max((len(row) for row in rows), default=0), ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return None, None, str(exc)


def audit(zip_path):
    groups = defaultdict(lambda: {"camera": set(), "lidar": set(), "behavior": None})
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            match = PATTERN.match(member)
            if not match:
                continue
            split, drive, relative = match.groups()
            key = (split.lower(), drive)
            item = groups[key]
            rel = Path(relative)
            stem = rel.stem
            if relative.lower() == "behavior.csv":
                item["behavior"] = member
            elif stem.isdigit() and relative.lower().startswith("dvr_") and rel.suffix.lower() in IMAGE_EXT:
                item["camera"].add(int(stem))
            elif stem.isdigit() and relative.lower().startswith("points_") and rel.suffix.lower() in LIDAR_EXT:
                item["lidar"].add(int(stem))
        results = []
        for (split, drive), item in sorted(groups.items()):
            camera, lidar = item["camera"], item["lidar"]
            nlabels, ncols, error = read_labels(archive, item["behavior"])
            results.append({
                "split": split, "drive_id": drive,
                "camera_count": len(camera), "lidar_count": len(lidar),
                "paired_frame_ids": len(camera & lidar),
                "camera_only_ids": len(camera - lidar),
                "lidar_only_ids": len(lidar - camera),
                "behavior_rows": nlabels, "behavior_columns": ncols,
                "label_read_error": error,
                "camera_lidar_ids_match": camera == lidar,
                "labels_match_camera_count": nlabels == len(camera) if nlabels is not None else False,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/article4"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = audit(args.archive)
    if not records:
        raise SystemExit("No train/val/test drives detected; inspect ZIP layout")
    with (args.output / "dbnet_full_drive_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        summary[split] = {
            "drives": len(subset),
            "camera_frames": sum(r["camera_count"] for r in subset),
            "lidar_frames": sum(r["lidar_count"] for r in subset),
            "paired_frames": sum(r["paired_frame_ids"] for r in subset),
            "label_rows": sum(r["behavior_rows"] or 0 for r in subset),
        }
    (args.output / "dbnet_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
