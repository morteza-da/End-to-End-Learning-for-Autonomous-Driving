"""Stable temporal membership fingerprints with no sensor file access."""
import unittest
from autonomous_driving.dataset_fingerprint import fingerprint_windows
from autonomous_driving.sequence_manifest import Observation


def window(drive, start):
    return tuple(Observation(
        sample_id=f"val/{drive}/{i}", drive_id=f"val/{drive}",
        frame_index=i, split="val", camera_path=f"val/{drive}/cam/{i}.jpg",
        lidar_path=f"val/{drive}/lidar/{i}.las",
        steering_deg=0.0, speed_kmh=0.0
    ) for i in (start, start + 1))


class FingerprintTests(unittest.TestCase):
    def test_order_independent_window_membership(self):
        a, b = window("a", 0), window("b", 0)
        self.assertEqual(fingerprint_windows([a, b]), fingerprint_windows([b, a]))

    def test_detects_changed_window_membership(self):
        self.assertNotEqual(
            fingerprint_windows([window("a", 0)]),
            fingerprint_windows([window("a", 1)])
        )

    def test_excludes_label_values(self):
        from dataclasses import replace
        original = window("a", 0)
        changed = tuple(replace(row, steering_deg=99.0) for row in original)
        self.assertEqual(fingerprint_windows([original]), fingerprint_windows([changed]))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            fingerprint_windows([])


if __name__ == "__main__":
    unittest.main()
