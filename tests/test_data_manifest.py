"""Synthetic unit tests only; not validation of real driving datasets."""
import csv
from pathlib import Path
import tempfile
import unittest

from autonomous_driving.data_manifest import validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "points").mkdir()
        (self.root / "images" / "a.png").write_bytes(b"fixture")
        (self.root / "points" / "a.bin").write_bytes(b"fixture")
        self.manifest = self.root / "manifest.csv"

    def write_rows(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sample_id", "split", "camera_path", "lidar_path", "steering", "speed"
            ])
            writer.writeheader()
            writer.writerows(rows)

    def row(self, **overrides):
        item = dict(sample_id="sample-1", split="train", camera_path="images/a.png",
                    lidar_path="points/a.bin", steering="0.5", speed="12.0")
        item.update(overrides)
        return item

    def test_valid_manifest(self):
        self.write_rows([self.row()])
        self.assertEqual(validate_manifest(self.manifest, self.root),
                         {"test": 0, "train": 1, "val": 0})

    def test_duplicate_id(self):
        self.write_rows([self.row(), self.row(split="val")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(self.manifest)

    def test_missing_sensor_file(self):
        self.write_rows([self.row(lidar_path="points/missing.bin")])
        with self.assertRaisesRegex(ValueError, "missing or unsafe"):
            validate_manifest(self.manifest, self.root)

    def test_nonfinite_label(self):
        self.write_rows([self.row(speed="nan")])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_manifest(self.manifest)

    def test_path_traversal(self):
        self.write_rows([self.row(camera_path="../secret.png")])
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
