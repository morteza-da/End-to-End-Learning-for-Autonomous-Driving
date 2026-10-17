import unittest

try:
    import numpy as np
except ImportError:
    np = None

from autonomous_driving.lidar_projection import RangeImageConfig, project_xyz_to_rgb


@unittest.skipUnless(np is not None, "numpy not installed")
class LiDARProjectionTests(unittest.TestCase):
    def test_empty_cloud(self):
        image, mask = project_xyz_to_rgb(np.empty((0, 3)))
        self.assertEqual(image.shape, (299, 299, 3))
        self.assertEqual(int(mask.sum()), 0)

    def test_nearest_return_wins(self):
        config = RangeImageConfig(width=10, height=10)
        points = np.array([[10., 0., 0.], [30., 0., 0.]])
        image, mask = project_xyz_to_rgb(points, config)
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(tuple(image[3, 5]), (0, 0, 128))

    def test_distance_color_bands(self):
        points = np.array([[10., 0., 0.], [0., 30., 0.], [-50., 0., 0.]])
        image, mask = project_xyz_to_rgb(points)
        colors = [tuple(v) for v in image[mask]]
        self.assertIn((0, 0, 128), colors)
        self.assertIn((0, 191, 0), colors)
        self.assertIn((212, 0, 0), colors)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            project_xyz_to_rgb(np.array([[float("nan"), 0., 0.]]))

    def test_field_of_view_filter(self):
        cfg = RangeImageConfig(elevation_min_deg=-5, elevation_max_deg=5)
        image, mask = project_xyz_to_rgb(np.array([[1., 0., 10.]]), cfg)
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
