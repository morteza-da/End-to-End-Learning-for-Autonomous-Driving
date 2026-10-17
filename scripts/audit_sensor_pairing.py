"""Read-only audit of DBNet camera/LiDAR frame pairing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autonomous_driving.pairing_audit import audit_pairing_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    reports = audit_pairing_dataset(args.dataset_root)
    result = {
        "audit_scope": "numeric frame IDs, file presence, nonzero size, contiguous runs",
        "timestamp_synchronization_verified": False,
        "calibration_verified": False,
        "drive_count": len(reports),
        "paired_frames": sum(x.paired_count for x in reports),
        "drives": [x.as_dict() for x in reports],
    }
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
