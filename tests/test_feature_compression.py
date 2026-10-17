"""Spatial fusion tests; no pretrained model downloads."""
import unittest

from autonomous_driving.feature_compression import compressed_feature_width

try:
    import torch
except ImportError:
    torch = None


class FeatureWidthTests(unittest.TestCase):
    def test_width_from_measured_shapes(self):
        a = (1, 2048, 10, 10)
        self.assertEqual(compressed_feature_width(a, a, pool_size=1), 4096)
        self.assertEqual(compressed_feature_width(a, a, pool_size=2), 16384)
        self.assertEqual(compressed_feature_width(a, a, pool_size=5), 102400)

    def test_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            compressed_feature_width((1, 2, 3), (1, 2, 3))
        with self.assertRaises(ValueError):
            compressed_feature_width((1, 2, 3, 4), (2, 2, 3, 4))
        with self.assertRaises(ValueError):
            compressed_feature_width((1, 2, 3, 4), (1, 2, 3, 4), pool_size=0)


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class FeaturePoolingTests(unittest.TestCase):
    def test_gradient_and_width(self):
        from autonomous_driving.feature_compression import pool_and_concatenate
        a = torch.randn(2, 3, 10, 10, requires_grad=True)
        b = torch.randn(2, 5, 10, 10, requires_grad=True)
        fused = pool_and_concatenate(a, b, pool_size=2)
        self.assertEqual(tuple(fused.shape), (2, 32))
        fused.square().sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)


if __name__ == "__main__":
    unittest.main()
