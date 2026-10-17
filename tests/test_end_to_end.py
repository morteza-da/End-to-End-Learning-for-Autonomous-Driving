"""Synthetic integration checks for the multimodal model interface."""
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class EndToEndTests(unittest.TestCase):
    def test_forward_backward(self):
        from autonomous_driving.end_to_end import (
            build_end_to_end_model, infer_feature_width, run_multimodal_epoch
        )

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 2, kernel_size=1),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

            def forward(self, x):
                return self.features(x)

        camera, lidar = TinyBackbone(), TinyBackbone()
        width = infer_feature_width(camera, lidar)
        self.assertEqual(width, 4)
        model = build_end_to_end_model(camera, lidar, input_size=width, hidden_size=4)
        camera_seq = torch.rand(1, 2, 3, 299, 299)
        lidar_seq = torch.rand(1, 2, 3, 299, 299)
        targets = torch.tensor([[2.0, 30.0]])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = run_multimodal_epoch(
            model, [(camera_seq, lidar_seq, targets)], optimizer=optimizer
        )
        self.assertTrue(0 <= loss < float("inf"))
        with torch.no_grad():
            prediction = model(camera_seq, lidar_seq)
        self.assertEqual(tuple(prediction["attention_weights"].shape), (1, 2))

    def test_pooled_variant_forward_backward(self):
        from autonomous_driving.end_to_end import (
            build_end_to_end_model, infer_feature_width
        )

        class TinySpatialBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 2, kernel_size=1),
                    nn.AdaptiveAvgPool2d((4, 4)),
                )

            def forward(self, x):
                return self.features(x)

        camera, lidar = TinySpatialBackbone(), TinySpatialBackbone()
        original = infer_feature_width(camera, lidar)
        pooled = infer_feature_width(camera, lidar, pool_size=2)
        self.assertEqual(original, 64)
        self.assertEqual(pooled, 16)
        model = build_end_to_end_model(
            camera, lidar, input_size=pooled, hidden_size=4, pool_size=2
        )
        frames = torch.rand(1, 2, 3, 299, 299)
        prediction = model(frames, frames)
        self.assertEqual(tuple(prediction["attention_weights"].shape), (1, 2))
        prediction["steering_deg"].sum().backward()
        self.assertIsNotNone(camera.features[0].weight.grad)
        self.assertIsNotNone(lidar.features[0].weight.grad)

    def test_invalid_sequence_length(self):
        from autonomous_driving.end_to_end import build_end_to_end_model
        model = build_end_to_end_model(nn.Identity(), nn.Identity(), input_size=6)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 1, 3, 299, 299),
                  torch.zeros(1, 1, 3, 299, 299))


if __name__ == "__main__":
    unittest.main()
