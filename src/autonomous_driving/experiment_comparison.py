"""Compare measured, schema-compatible validation reports without inventing scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_comparison_rows(paths):
    """Load genuine validation reports and reject incompatible experiments."""
    rows = []
    baseline = None
    for item in paths:
        path = Path(item)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("split") != "val":
            raise ValueError(f"{path}: expected validation split")
        config = report.get("config")
        metrics = report.get("metrics")
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            raise ValueError(f"{path}: missing experiment config or measured metrics")
        required = ("assignment", "pool_size", "seed", "pretrained",
                    "sequence_length", "stride", "batch_size", "epochs",
                    "learning_rate", "hidden_size", "lstm_layers",
                    "camera_trainable_last_n", "lidar_trainable_last_n")
        if any(key not in config for key in required):
            raise ValueError(f"{path}: missing reproducibility settings")
        signature = (report.get("dataset_id"), tuple(
            (key, config[key]) for key in required if key != "pool_size"
        ))
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            raise ValueError(f"{path}: experiments have incompatible configurations")
        try:
            steering = metrics["steering_deg"]
            speed = metrics["speed_kmh"]
            threshold = metrics["threshold_accuracy_percent"]
            row = {
                "variant": "flatten" if config["pool_size"] is None else f"pool-{config['pool_size']}x{config['pool_size']}",
                "steering_rmse_deg": steering["rmse"],
                "steering_r2": steering["r2"],
                "speed_rmse_kmh": speed["rmse"],
                "speed_r2": speed["r2"],
                "steering_accuracy_6deg_percent": threshold["steering_accuracy_percent"],
                "speed_accuracy_5kmh_percent": threshold["speed_accuracy_percent"],
                "sample_count": metrics["sample_count"],
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: missing expected measured metrics") from exc
        if any(row[key] is None for key in row if key not in ("steering_r2", "speed_r2")):
            raise ValueError(f"{path}: missing measured values")
        if not isinstance(row["sample_count"], int) or row["sample_count"] < 1:
            raise ValueError(f"{path}: invalid sample count")
        rows.append(row)
    if not rows:
        raise ValueError("Supply at least one validation report")
    if len({row["sample_count"] for row in rows}) != 1:
        raise ValueError("Validation sample counts differ")
    if len({row["variant"] for row in rows}) != len(rows):
        raise ValueError("Duplicate fusion variants")
    return sorted(rows, key=lambda row: (row["variant"] != "flatten", row["variant"]))


def write_comparison_csv(rows, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
