"""Lightweight CLI contract checks; no model weights or dataset required."""
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_multimodal.py"


class TrainingCLIContractTests(unittest.TestCase):
    def test_help_exposes_stride_and_integrity_flags(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stride", result.stdout)
        self.assertIn("--require-sensor-hashes", result.stdout)
        self.assertIn("--preflight-report", result.stdout)

    def test_invalid_stride_is_rejected_by_experiment_config(self):
        from autonomous_driving.experiment_config import ExperimentConfig
        with self.assertRaises(ValueError):
            ExperimentConfig(stride=0)

    def test_valid_stride_is_saved_in_config(self):
        from autonomous_driving.experiment_config import ExperimentConfig
        self.assertEqual(ExperimentConfig(stride=2).as_dict()["stride"], 2)


if __name__ == "__main__":
    unittest.main()
