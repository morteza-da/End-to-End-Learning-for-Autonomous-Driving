import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

from autonomous_driving.dbnet_media import load_camera


@unittest.skipUnless(np is not None and Image is not None, "Pillow/numpy not installed")
class DBNetMediaTests(unittest.TestCase):
    def test_rgb_resize_and_channel_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0.jpg"
            Image.new("RGB", (200, 66), (100, 20, 5)).save(path)
            result = load_camera(path)
            self.assertEqual(result.shape, (3, 299, 299))
            self.assertTrue(np.isfinite(result).all())
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
