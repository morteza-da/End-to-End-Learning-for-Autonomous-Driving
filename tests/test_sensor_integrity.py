"""Sensor content integrity checks using small synthetic files."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autonomous_driving.sensor_integrity import fingerprint_sensor_files


class SensorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "camera.jpg").write_bytes(b"camera-one")
        (self.root / "cloud.las").write_bytes(b"lidar-one")
        self.rows = [SimpleNamespace(camera_path="camera.jpg", lidar_path="cloud.las")]

    def test_stable(self):
        first = fingerprint_sensor_files(self.root, self.rows)
        self.assertEqual(first, fingerprint_sensor_files(self.root, self.rows))
        self.assertEqual(first["file_count"], 2)

    def test_detects_modified_bytes(self):
        first = fingerprint_sensor_files(self.root, self.rows)
        (self.root / "cloud.las").write_bytes(b"lidar-two")
        self.assertNotEqual(first, fingerprint_sensor_files(self.root, self.rows))

    def test_missing_file(self):
        (self.root / "camera.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            fingerprint_sensor_files(self.root, self.rows)

    def test_reject_path_escape(self):
        rows = [SimpleNamespace(camera_path="../outside.jpg", lidar_path="cloud.las")]
        with self.assertRaises(ValueError):
            fingerprint_sensor_files(self.root, rows)


if __name__ == "__main__":
    unittest.main()
