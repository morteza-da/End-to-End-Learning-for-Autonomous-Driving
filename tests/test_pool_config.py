"""Ensure fusion variants are retained in reproducible experiment manifests."""
import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.experiment_config import (
    ExperimentConfig, load_experiment_config, save_experiment_config
)


class PoolConfigTests(unittest.TestCase):
    def test_baseline_default(self):
        self.assertIsNone(ExperimentConfig().pool_size)

    def test_round_trip_pooled_variant(self):
        config = ExperimentConfig(assignment="B", pool_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_experiment_config(config, path)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["pool_size"], 2)
            self.assertEqual(load_experiment_config(path), config)

    def test_reject_invalid_pool_size(self):
        for value in (0, 3, -1, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ExperimentConfig(pool_size=value)


if __name__ == "__main__":
    unittest.main()
