"""Comparison tables must be grounded in measured, compatible reports."""
import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.experiment_comparison import (
    load_comparison_rows, write_comparison_csv
)
from autonomous_driving.experiment_config import ExperimentConfig


def report(pool_size, *, samples=4, seed=42, split="val"):
    return {
        "dataset_id": "DBNet-2018",
        "split": split,
        "config": ExperimentConfig(pool_size=pool_size, seed=seed).as_dict(),
        "metrics": {
            "steering_deg": {"rmse": 2.5, "r2": 0.8},
            "speed_kmh": {"rmse": 3.2, "r2": None},
            "threshold_accuracy_percent": {
                "steering_accuracy_percent": 75.0,
                "speed_accuracy_percent": 50.0,
            },
            "sample_count": samples,
        },
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def save(self, name, payload):
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_two_variants_csv(self):
        a = self.save("flatten.json", report(None))
        b = self.save("pool.json", report(2))
        rows = load_comparison_rows([b, a])
        self.assertEqual([r["variant"] for r in rows], ["flatten", "pool-2x2"])
        self.assertIsNone(rows[0]["speed_r2"])
        destination = self.directory / "comparison.csv"
        write_comparison_csv(rows, destination)
        self.assertIn("steering_rmse_deg", destination.read_text())

    def test_reject_mismatched_seed(self):
        a = self.save("a.json", report(None))
        b = self.save("b.json", report(2, seed=123))
        with self.assertRaises(ValueError):
            load_comparison_rows([a, b])

    def test_reject_mismatched_samples(self):
        a = self.save("a.json", report(None))
        b = self.save("b.json", report(2, samples=3))
        with self.assertRaises(ValueError):
            load_comparison_rows([a, b])

    def test_reject_nonvalidation(self):
        a = self.save("a.json", report(None, split="train+val"))
        with self.assertRaises(ValueError):
            load_comparison_rows([a])


if __name__ == "__main__":
    unittest.main()
