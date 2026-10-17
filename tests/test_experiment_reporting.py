import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.experiment_config import (
    ExperimentConfig, load_experiment_config, save_experiment_config,
)
from autonomous_driving.experiment_reporting import (
    evaluate_driving_predictions, write_experiment_report,
)


class ExperimentTests(unittest.TestCase):
    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config = ExperimentConfig(assignment="B", sequence_length=6)
            save_experiment_config(config, path)
            self.assertEqual(load_experiment_config(path), config)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(sequence_length=1)
        with self.assertRaises(ValueError):
            ExperimentConfig(assignment="C")

    def test_metrics_and_report(self):
        metrics = evaluate_driving_predictions(
            [0.0, 10.0], [1.0, 9.0],
            [20.0, 30.0], [22.0, 29.0]
        )
        self.assertEqual(metrics["sample_count"], 2)
        self.assertAlmostEqual(metrics["steering_deg"]["mae"], 1.0)
        self.assertEqual(
            metrics["threshold_accuracy_percent"]["steering_accuracy_percent"],
            100.0
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            write_experiment_report(
                path, config=ExperimentConfig(), metrics=metrics,
                dataset_id="fixture", split="val", notes="synthetic test only"
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset_id"], "fixture")
            self.assertEqual(saved["metrics"]["sample_count"], 2)

    def test_reject_nonfinite_report_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_experiment_report(
                    Path(folder) / "report.json",
                    config=ExperimentConfig(), metrics={"rmse": float("nan")},
                    dataset_id="fixture", split="val"
                )


if __name__ == "__main__":
    unittest.main()
