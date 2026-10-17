"""Tests use synthetic fixtures, not dissertation experimental results."""
import math
import unittest

from autonomous_driving.metrics import regression_metrics


class RegressionMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(result, {"rmse": 0.0, "r2": 1.0})

    def test_known_error(self):
        result = regression_metrics([0, 2], [1, 1])
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["r2"], 0.0)

    def test_constant_target_r2_undefined(self):
        self.assertTrue(math.isnan(regression_metrics([1, 1], [1, 1])["r2"]))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            regression_metrics([], [])
        with self.assertRaises(ValueError):
            regression_metrics([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
