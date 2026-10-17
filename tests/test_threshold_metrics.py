"""Synthetic-only tests for dissertation-defined threshold accuracy."""
import math
import unittest

from autonomous_driving.threshold_metrics import (
    driving_threshold_accuracies,
    threshold_accuracy,
)


class ThresholdAccuracyTests(unittest.TestCase):
    def test_inclusive_boundary(self):
        self.assertEqual(threshold_accuracy([0, 0, 0], [6, -6, 6.01], 6), 200 / 3)

    def test_speed_and_steering_separately(self):
        result = driving_threshold_accuracies(
            steering_true_deg=[0, 0, 0, 0],
            steering_pred_deg=[6, -6, 6.1, 0],
            speed_true_kmh=[10, 10, 10, 10],
            speed_pred_kmh=[15, 4.9, 10, 5],
        )
        self.assertEqual(result["steering_accuracy_percent"], 75.0)
        self.assertEqual(result["speed_accuracy_percent"], 75.0)

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            threshold_accuracy([], [], 6)

    def test_reject_bad_threshold(self):
        for tolerance in [-1, math.nan, math.inf]:
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    threshold_accuracy([0], [0], tolerance)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            threshold_accuracy([math.nan], [0], 6)

    def test_reject_mismatched_outputs(self):
        with self.assertRaises(ValueError):
            driving_threshold_accuracies([0], [0], [1, 2], [1, 2])


if __name__ == "__main__":
    unittest.main()
