import unittest
from autonomous_driving.sequence_manifest import Observation, build_sequence_manifest, split_windows


def row(drive, index, split="train"):
    return Observation(f"{drive}-{index}", drive, index, split, "camera.png",
                       "lidar.png", 0.0, 10.0)


class SequenceManifestTests(unittest.TestCase):
    def test_consecutive_windows(self):
        windows = build_sequence_manifest([row("d1", i) for i in range(5)],
                                          sequence_length=3)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0][-1].frame_index, 2)

    def test_gap_does_not_bridge(self):
        windows = build_sequence_manifest([row("d1", i) for i in (0, 1, 3, 4)],
                                          sequence_length=2)
        self.assertEqual(len(windows), 2)

    def test_no_cross_drive(self):
        windows = build_sequence_manifest([row("a", i) for i in range(3)] +
                                          [row("b", i, "test") for i in range(3)],
                                          sequence_length=2)
        self.assertEqual({k: len(v) for k, v in split_windows(windows).items()},
                         {"train": 2, "val": 0, "test": 2})

    def test_reject_drive_leakage(self):
        with self.assertRaises(ValueError):
            build_sequence_manifest([row("d", 0, "train"),
                                     row("d", 1, "test")], sequence_length=2)

    def test_reject_duplicate_index(self):
        with self.assertRaises(ValueError):
            build_sequence_manifest([row("d", 0), row("d", 0)],
                                    sequence_length=2)


if __name__ == "__main__":
    unittest.main()
