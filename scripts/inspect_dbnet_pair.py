"""Decode one DBNet sensor pair without assuming label column units.

Usage: python scripts/inspect_dbnet_pair.py data/dbnet-2018/train/1 0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from autonomous_driving.dbnet_media import verify_frame_pair
from autonomous_driving.dbnet_layout import read_behavior_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("drive_folder", type=Path)
    parser.add_argument("frame_index", type=int)
    args = parser.parse_args()
    drive, index = args.drive_folder, args.frame_index
    report = verify_frame_pair(drive / "dvr_66x200" / f"{index}.jpg",
                               drive / "points_16384" / f"{index}.las")
    labels = drive / "behavior.csv"
    if labels.is_file():
        rows = read_behavior_rows(labels)
        report["raw_label_columns"] = rows[index] if index < len(rows) else None
    else:
        report["raw_label_columns"] = None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
