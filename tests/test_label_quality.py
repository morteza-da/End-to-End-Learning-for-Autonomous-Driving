"""Synthetic checks for physical target-range audit."""
import unittest

from autonomous_driving.label_quality import audit_target_ranges, require_plausible_targets
from autonomous_driving.sequence_manifest import Observation


def row(i, steering, speed, split="train"):
    return Observation(
        sample_id=f"{split}/0/{i}", drive_id=f"{split}/0", frame_index=i,
        split=split, camera_path=f"{i}.jpg", lidar_path=f"{i}.las",
        steering_deg=steering, speed_kmh=speed
    )


class TargetQualityTests(unittest.TestCase):
    def test_valid_boundaries(self):
        result = audit_target_ranges([row(0, -40, 0), row(1, 40, 120)])
        self.assertEqual(result["train"]["steering_out_of_range"], 0)
        self.assertEqual(result["train"]["speed_out_of_range"], 0)
        require_plausible_targets(result)

    def test_reject_out_of_range(self):
        result = audit_target_ranges([row(0, 40.01, 121)])
        self.assertEqual(result["train"]["steering_out_of_range"], 1)
        self.assertEqual(result["train"]["speed_out_of_range"], 1)
        with self.assertRaises(ValueError):
            require_plausible_targets(result)

    def test_reject_negative_speed(self):
        with self.assertRaises(ValueError):
            require_plausible_targets(audit_target_ranges([row(0, 0, -0.01)]))

    def test_reject_nonfinite_or_nonpositive_limits(self):
        for steering_limit, speed_limit in (
            (float("nan"), 120), (float("inf"), 120),
            (0, 120), (-1, 120), (40, float("nan")),
            (40, float("inf")), (40, 0), (40, -1),
        ):
            with self.subTest(steering_limit=steering_limit, speed_limit=speed_limit):
                with self.assertRaisesRegex(ValueError, "Limits"):
                    audit_target_ranges(
                        [row(0, 0, 10)],
                        steering_limit_deg=steering_limit,
                        speed_limit_kmh=speed_limit,
                    )

    def test_custom_limits_are_applied_without_clipping(self):
        observations = [row(0, -180, 227), row(1, 20, -4)]
        result = audit_target_ranges(
            observations, steering_limit_deg=180, speed_limit_kmh=230
        )
        self.assertEqual(result["train"]["steering_deg"]["min"], -180)
        self.assertEqual(result["train"]["speed_kmh"]["max"], 227)
        self.assertEqual(result["train"]["speed_out_of_range"], 1)
        with self.assertRaises(ValueError):
            require_plausible_targets(result)

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            audit_target_ranges([])


if __name__ == "__main__":
    unittest.main()
