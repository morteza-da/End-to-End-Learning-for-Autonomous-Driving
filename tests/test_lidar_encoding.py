"""Synthetic tests for the documented LiDAR color-encoding interpretation."""
import math
import unittest

from autonomous_driving.lidar_encoding import (
    distance_to_rgb,
    encode_points_xy,
    planar_distance,
)


class LiDAREncodingTests(unittest.TestCase):
    def test_planar_distance_ignores_z_by_design(self):
        self.assertEqual(planar_distance(3, 4), 5)

    def test_blue_below_twenty(self):
        self.assertEqual(distance_to_rgb(10), (0, 0, 128))

    def test_green_at_twenty(self):
        self.assertEqual(distance_to_rgb(20), (0, 128, 0))

    def test_green_below_forty(self):
        self.assertEqual(distance_to_rgb(30), (0, 191, 0))

    def test_red_at_forty(self):
        self.assertEqual(distance_to_rgb(40), (170, 0, 0))

    def test_red_clamps_at_sixty(self):
        self.assertEqual(distance_to_rgb(60), (255, 0, 0))
        self.assertEqual(distance_to_rgb(100), (255, 0, 0))

    def test_zero_distance(self):
        self.assertEqual(distance_to_rgb(0), (0, 0, 0))

    def test_points_not_pixels(self):
        self.assertEqual(encode_points_xy([(0, 10), (3, 4)]),
                         [(0, 0, 128), (0, 0, 64)])

    def test_reject_nonfinite_and_negative(self):
        for value in (-1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    distance_to_rgb(value)
        with self.assertRaises(ValueError):
            planar_distance(math.inf, 1)


if __name__ == "__main__":
    unittest.main()
