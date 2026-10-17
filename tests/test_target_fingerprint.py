"""Synthetic tests for canonical target fingerprints."""
import unittest
from dataclasses import replace

from autonomous_driving.sequence_manifest import Observation
from autonomous_driving.target_fingerprint import fingerprint_targets


class TargetFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            Observation("train/d/0", "train/d", 0, "train",
                        "camera/0.jpg", "lidar/0.las", 1.0, 25.0),
            Observation("train/d/1", "train/d", 1, "train",
                        "camera/1.jpg", "lidar/1.las", 2.0, 30.0),
        ]

    def test_order_independent(self):
        self.assertEqual(fingerprint_targets(self.rows),
                         fingerprint_targets(reversed(self.rows)))

    def test_changed_speed_changes_hash(self):
        modified = [self.rows[0], replace(self.rows[1], speed_kmh=31.0)]
        self.assertNotEqual(fingerprint_targets(self.rows),
                            fingerprint_targets(modified))

    def test_changed_steering_changes_hash(self):
        modified = [replace(self.rows[0], steering_deg=3.0), self.rows[1]]
        self.assertNotEqual(fingerprint_targets(self.rows),
                            fingerprint_targets(modified))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            fingerprint_targets([])


if __name__ == "__main__":
    unittest.main()
