#!/usr/bin/env python3
"""Build four-frame, stride-four validation windows directly from DBNet ZIP.

Keeps original frame IDs. Uses the label at the final frame in each window.
Only constructs a manifest; it does NOT run inference or verify sensor timestamps.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile


def make_manifest(archive, split="val", sequence_length=4, stride=4):
    records = []
    with ZipFile(archive) as z:
        names = set(z.namelist())
        drives = sorted({p.split("/")[1] for p in names
                         if p.startswith(split + "/") and len(p.split("/")) > 2
                         and p.split("/")[1].isdigit()}, key=int)
        for drive in drives:
            prefix = f"{split}/{drive}/"
            camera, lidar = {}, {}
            for name in names:
                if not name.startswith(prefix):
                    continue
                rel = name[len(prefix):]
                file = Path(rel)
                if not file.stem.isdigit():
                    continue
                index = int(file.stem)
                if rel.startswith("dvr_") and file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    camera[index] = name
                if rel.startswith("points_") and file.suffix.lower() in {".las", ".laz"}:
                    lidar[index] = name
            label_member = prefix + "behavior.csv"
            if label_member not in names:
                continue
            labels = list(csv.reader(io.TextIOWrapper(z.open(label_member), encoding="utf-8-sig")))
            paired = set(camera) & set(lidar)
            for start in range(0, len(labels) - sequence_length + 1, stride):
                ids = list(range(start, start + sequence_length))
                if not all(i in paired for i in ids):
                    continue
                target = labels[ids[-1]]
                if len(target) < 2:
                    continue
                try:
                    steering, speed = float(target[0]), float(target[1])
                except ValueError:
                    continue
                records.append({
                    "split": split, "drive_id": drive,
                    "window_start": start, "window_end": ids[-1],
                    "frame_ids": ",".join(map(str, ids)),
                    "camera_members": "|".join(camera[i] for i in ids),
                    "lidar_members": "|".join(lidar[i] for i in ids),
                    "steering_true_deg": steering, "speed_true_kmh": speed,
                })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/article4"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.stride < 1:
        parser.error("sequence-length and stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = make_manifest(args.archive, args.split, args.sequence_length, args.stride)
    if not records:
        raise SystemExit("No labeled, fully paired windows found")
    destination = args.output / f"article4_{args.split}_window_manifest.csv"
    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"windows": len(records), "drives": len(set(r["drive_id"] for r in records)), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
