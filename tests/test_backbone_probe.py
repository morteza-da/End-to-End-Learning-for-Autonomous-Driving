"""Probe contract tests with tiny local PyTorch backbones."""
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class BackboneProbeTests(unittest.TestCase):
    def test_actual_shapes_drive_width(self):
        from autonomous_driving.backbone_probe import probe_backbones

        class TinyBackbone(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.layer = nn.Conv2d(3, channels, kernel_size=1)
                self.pool = nn.AdaptiveAvgPool2d((2, 3))

            def forward(self, x):
                return self.pool(self.layer(x))

        def fake_build(architecture, *, pretrained, names):
            return TinyBackbone(2 if architecture == "xception" else 3)

        with patch("autonomous_driving.backbone_probe.build_backbone", side_effect=fake_build):
            result = probe_backbones("A", hidden_size=8)
        self.assertEqual(result.fused_feature_width, 30)
        self.assertEqual(result.camera_shape[0], 1)
        self.assertEqual(result.lidar_shape[0], 1)
        self.assertGreater(result.temporal_parameters, 0)
        self.assertGreater(result.temporal_adam_state_mib, 0)

    def test_invalid_assignment(self):
        from autonomous_driving.backbone_probe import probe_backbones
        with self.assertRaises(ValueError):
            probe_backbones("invalid")


if __name__ == "__main__":
    unittest.main()
