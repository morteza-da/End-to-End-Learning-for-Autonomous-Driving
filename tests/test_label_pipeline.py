import math
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.label_inspection import (
    extract_raw_frame_labels, inspect_behavior_csv,
)
from autonomous_driving.target_mapping import TargetSchema, map_raw_observations


class LabelPipelineTests(unittest.TestCase):
    def test_channel_major_two_by_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behavior.csv"
            path.write_text("1,2,3\n10,20,30\n", encoding="utf-8")
            report = inspect_behavior_csv(path, expected_frames=3)
            self.assertEqual(report["orientation"], "frames_as_columns")
            self.assertEqual(
                extract_raw_frame_labels(
                    path, frame_count=3, orientation="frames_as_columns"
                ),
                [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
            )

    def test_explicit_square_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behavior.csv"
            path.write_text("1,2\n3,4\n", encoding="utf-8")
            self.assertEqual(
                inspect_behavior_csv(path, expected_frames=2)["orientation"],
                "ambiguous"
            )
            self.assertEqual(
                extract_raw_frame_labels(
                    path, frame_count=2, orientation="frames_as_columns"
                ),
                [(1.0, 3.0), (2.0, 4.0)]
            )

    def test_convert_verified_units(self):
        record = {
            "sample_id": "train/1/0",
            "drive_id": "train/1",
            "frame_index": 0,
            "split": "train",
            "camera_path": "train/1/dvr_66x200/0.jpg",
            "lidar_path": "train/1/points_16384/0.las",
            "raw_label_columns": (10.0, math.pi / 2),
        }
        schema = TargetSchema(
            steering_channel=1, speed_channel=0,
            steering_unit="rad", speed_unit="m/s"
        )
        result = map_raw_observations([record], schema)
        self.assertAlmostEqual(result[0].steering_deg, 90.0)
        self.assertAlmostEqual(result[0].speed_kmh, 36.0)

    def test_reject_invalid_schema(self):
        with self.assertRaises(ValueError):
            TargetSchema(0, 0, "deg", "km/h")


if __name__ == "__main__":
    unittest.main()
