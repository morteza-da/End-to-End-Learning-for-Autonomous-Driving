"""Run: python scripts/audit_dbnet.py /path/to/extracted/dbnet-2018"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from autonomous_driving.dbnet_layout import audit_dataset


def main():
    parser = argparse.ArgumentParser(description="Audit extracted DBNet folders")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional local JSON audit output")
    args = parser.parse_args()
    audits = audit_dataset(args.dataset_root)
    payload = {
        "dataset_root": str(args.dataset_root),
        "drives": [asdict(x) for x in audits],
        "totals": {
            split: {
                "drives": sum(x.split == split for x in audits),
                "paired_frames": sum(x.paired_count for x in audits if x.split == split),
                "labeled_rows": sum(x.label_rows or 0 for x in audits if x.split == split),
            } for split in ("train", "val", "test")
        },
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
