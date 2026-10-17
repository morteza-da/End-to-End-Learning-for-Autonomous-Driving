"""Tests for the public tracked-file policy."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_public_files import unsafe_tracked_path


class PublicFilePolicyTests(unittest.TestCase):
    def test_rejects_sensor_and_weight_files(self):
        for path in ("data/train/0.jpg", "datasets/a.las", "weights/model.pt",
                     "results/report.json", "checkpoints/epoch.ckpt",
                     ".env", ".env.local", "credentials.json"):
            with self.subTest(path=path):
                self.assertTrue(unsafe_tracked_path(path))

    def test_allows_source_and_placeholders(self):
        for path in ("README.md", "src/autonomous_driving/model.py",
                     "tests/test_public_file_policy.py", "data/.gitkeep",
                     "results/.gitkeep", "docs/example.json"):
            with self.subTest(path=path):
                self.assertFalse(unsafe_tracked_path(path))


if __name__ == "__main__":
    unittest.main()
