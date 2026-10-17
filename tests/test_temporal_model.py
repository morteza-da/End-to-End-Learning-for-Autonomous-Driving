"""Synthetic temporal model tests; skip cleanly when torch is unavailable."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from autonomous_driving.temporal_model import build_temporal_model


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class TemporalModelTests(unittest.TestCase):
    def test_output_shape_and_attention_normalization(self):
        model = build_temporal_model(input_size=12, hidden_size=8)
        out = model(torch.randn(2, 4, 12))
        self.assertEqual(tuple(out["steering_deg"].shape), (2,))
        self.assertEqual(tuple(out["speed_kmh"].shape), (2,))
        self.assertEqual(tuple(out["attention_weights"].shape), (2, 4))
        torch.testing.assert_close(out["attention_weights"].sum(dim=1),
                                   torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_gradient_flows_through_lstm(self):
        model = build_temporal_model(12, 8)
        out = model(torch.randn(2, 3, 12))
        (out["steering_deg"].square().mean() +
         out["speed_kmh"].square().mean()).backward()
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)

    def test_bad_input(self):
        model = build_temporal_model(12, 8)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 12))


if __name__ == "__main__":
    unittest.main()
