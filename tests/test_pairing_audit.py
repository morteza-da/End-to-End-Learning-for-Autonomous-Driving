"""Tests for index-level camera/LiDAR pairing audits."""
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.pairing_audit import audit_pairing_drive, audit_pairing_dataset


class PairingAuditTests(unittest.TestCase):
    def test_gaps_missing_modalities_and_zero_byte_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive = Path(tmp) / "drive-1"
            cam = drive / "dvr_66x200"
            lidar = drive / "points_16384"
            cam.mkdir(parents=True)
            lidar.mkdir()
            for i in (0, 1, 3, 4, 7):
                (cam / f"{i}.jpg").write_bytes(b"x" if i != 7 else b"")
            for i in (0, 1, 2, 3, 4):
                (lidar / f"{i}.las").write_bytes(b"x" if i != 2 else b"")
            result = audit_pairing_drive(drive, split="train")
            self.assertEqual(result.paired_count, 4)
            self.assertEqual(result.camera_only_ids, (7,))
            self.assertEqual(result.lidar_only_ids, (2,))
            self.assertEqual(result.zero_byte_camera_ids, (7,))
            self.assertEqual(result.zero_byte_lidar_ids, (2,))
            self.assertEqual(result.consecutive_pair_runs, ((0, 1), (3, 4)))

    def test_dataset_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                drive = root / split / "1"
                (drive / "dvr_66x200").mkdir(parents=True)
                (drive / "points_16384").mkdir()
                (drive / "dvr_66x200" / "0.jpg").write_bytes(b"x")
                (drive / "points_16384" / "0.las").write_bytes(b"x")
            result = audit_pairing_dataset(root)
            self.assertEqual(len(result), 3)
            self.assertEqual(sum(x.paired_count for x in result), 3)

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            audit_pairing_drive("unused", split="other")


if __name__ == "__main__":
    unittest.main()
