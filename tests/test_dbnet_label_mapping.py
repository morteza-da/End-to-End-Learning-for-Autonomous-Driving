"""Check raw DBNet column order and physical-unit conversions."""
import unittest

from autonomous_driving.target_mapping import TargetSchema, map_raw_observations


def raw_row(values):
    return [{
        "sample_id": "train/0/0",
        "drive_id": "train/0",
        "split": "train",
        "frame_index": 0,
        "camera_path": "train/0/dvr_66x200/0.jpg",
        "lidar_path": "train/0/points_16384/0.las",
        "raw_label_columns": values,
    }]


class DBNetLabelMappingTests(unittest.TestCase):
    def test_source_supported_column_order(self):
        schema = TargetSchema(
            steering_channel=0, speed_channel=1,
            steering_unit="deg", speed_unit="km/h"
        )
        row = map_raw_observations(raw_row(("12.5", "36.0")), schema)[0]
        self.assertEqual(row.steering_deg, 12.5)
        self.assertEqual(row.speed_kmh, 36.0)

    def test_explicit_units_convert_to_canonical_targets(self):
        schema = TargetSchema(0, 1, "rad", "m/s")
        row = map_raw_observations(raw_row(("0.5", "10")), schema)[0]
        self.assertAlmostEqual(row.steering_deg, 28.6478897565)
        self.assertAlmostEqual(row.speed_kmh, 36.0)


if __name__ == "__main__":
    unittest.main()
