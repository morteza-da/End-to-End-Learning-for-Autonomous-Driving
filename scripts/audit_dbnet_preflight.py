"""Read-only DBNet training preflight; does not train or copy sensor data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-provenance", type=Path, required=True)
    parser.add_argument("--steering-channel", type=int, required=True)
    parser.add_argument("--speed-channel", type=int, required=True)
    parser.add_argument("--steering-unit", choices=("deg", "rad"), required=True)
    parser.add_argument("--speed-unit", choices=("km/h", "m/s", "mph"), required=True)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--steering-limit-deg", type=float, default=40.0)
    parser.add_argument("--speed-limit-kmh", type=float, default=120.0)
    parser.add_argument("--hash-sensor-files", action="store_true",
                        help="Stream and hash all paired camera/LiDAR file bytes (slow)")
    parser.add_argument("--output", type=Path, default=Path("results/dbnet-preflight.json"))
    args = parser.parse_args()

    from autonomous_driving.dbnet_layout import audit_dataset, build_labeled_observations
    from autonomous_driving.target_mapping import TargetSchema, map_raw_observations
    from autonomous_driving.label_provenance import load_verified_label_provenance
    from autonomous_driving.label_quality import audit_target_ranges
    from autonomous_driving.sequence_manifest import build_sequence_manifest, split_windows
    from autonomous_driving.dataset_fingerprint import fingerprint_windows
    from autonomous_driving.target_fingerprint import fingerprint_targets
    from autonomous_driving.sensor_integrity import fingerprint_sensor_files
    from autonomous_driving.output_safety import require_output_outside_dataset

    require_output_outside_dataset(args.data_root, args.output)

    schema = TargetSchema(
        steering_channel=args.steering_channel,
        speed_channel=args.speed_channel,
        steering_unit=args.steering_unit,
        speed_unit=args.speed_unit,
    )
    evidence = load_verified_label_provenance(args.label_provenance, schema)
    drives = audit_dataset(args.data_root)
    raw = build_labeled_observations(args.data_root)
    observations = map_raw_observations(raw, schema)
    ranges = audit_target_ranges(
        observations, steering_limit_deg=args.steering_limit_deg,
        speed_limit_kmh=args.speed_limit_kmh,
    )
    windows = split_windows(build_sequence_manifest(
        observations, sequence_length=args.sequence_length, stride=args.stride
    ))
    report = {
        "dataset_root": str(args.data_root.resolve()),
        "provenance_source": evidence["source_url"],
        "target_schema": {
            "steering_channel": schema.steering_channel,
            "speed_channel": schema.speed_channel,
            "steering_unit": schema.steering_unit,
            "speed_unit": schema.speed_unit,
        },
        "plausibility_limits": {
            "steering_abs_deg": args.steering_limit_deg,
            "speed_min_kmh": 0,
            "speed_max_kmh": args.speed_limit_kmh,
        },
        "target_ranges": ranges,
        "canonical_targets_sha256": fingerprint_targets(observations),
        "sensor_content_integrity": (
            fingerprint_sensor_files(args.data_root, observations)
            if args.hash_sensor_files else None
        ),
        "drives": [
            {
                "split": d.split, "drive": d.drive,
                "camera_count": d.camera_count,
                "lidar_count": d.lidar_count,
                "paired_count": d.paired_count,
                "label_rows": d.label_rows,
                "missing_camera_count": len(d.missing_camera),
                "missing_lidar_count": len(d.missing_lidar),
                "missing_labels_count": len(d.missing_labels),
            } for d in drives
        ],
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "window_counts": {split: len(group) for split, group in windows.items()},
        "window_membership_sha256": {
            split: fingerprint_windows(group) if group else None
            for split, group in windows.items()
        },
        "readiness": {
            "train_and_val_nonempty": bool(windows["train"] and windows["val"]),
            "no_target_range_violations": all(
                not values["steering_out_of_range"] and not values["speed_out_of_range"]
                for values in ranges.values()
            ),
            "no_missing_sensor_or_label_files": all(
                not (d.missing_camera or d.missing_lidar or d.missing_labels)
                for d in drives
            ),
        },
        "note": "Frame-index pairing does not prove physical timestamp synchronization.",
    }
    report["readiness"]["passed"] = all(report["readiness"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Preflight report: {args.output}")
    print(f"Preflight passed: {report['readiness']['passed']}")
    if not report["readiness"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
