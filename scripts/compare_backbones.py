"""Compare measured A/B validation metrics without manufacturing results."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def compare_reports(path_a, path_b):
    reports = {}
    for name, path in (("A", path_a), ("B", path_b)):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report["config"]["assignment"] != name:
            raise ValueError(f"Expected assignment {name} in {path}")
        if report.get("dataset_id") != "DBNet-2018":
            raise ValueError("Both reports must use DBNet-2018")
        if report.get("split") != "train+val":
            raise ValueError("Both reports must contain train+val objectives")
        metrics = report["metrics"]
        if "best_validation_objective" not in metrics:
            raise ValueError(f"Missing validation objective in {path}")
        value = float(metrics["best_validation_objective"])
        if not math.isfinite(value):
            raise ValueError("Validation objective must be finite")
        reports[name] = {"best_validation_objective": value,
                         "config": report["config"],
                         "dataset_id": report["dataset_id"]}
    a, b = reports["A"], reports["B"]
    for key in ("sequence_length", "stride", "batch_size", "epochs",
                "learning_rate", "seed", "steering_weight", "speed_weight"):
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"Noncomparable A/B configurations: {key}")
    winner = "A" if a["best_validation_objective"] < b["best_validation_objective"] else (
        "B" if b["best_validation_objective"] < a["best_validation_objective"] else "tie"
    )
    return {"A": a["best_validation_objective"],
            "B": b["best_validation_objective"],
            "lower_validation_objective": winner,
            "note": "Validation training objective only; not held-out test performance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison = compare_reports(args.report_a, args.report_b)
    print(json.dumps(comparison, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
