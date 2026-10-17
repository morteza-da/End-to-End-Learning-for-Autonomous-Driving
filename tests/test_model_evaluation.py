import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from autonomous_driving.model_evaluation import evaluate_driving_loader


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class ModelEvaluationTests(unittest.TestCase):
    def test_aggregates_all_batches(self):
        class ConstantModel(nn.Module):
            def forward(self, camera, lidar):
                batch = camera.shape[0]
                return {
                    "steering_deg": torch.zeros(batch),
                    "speed_kmh": torch.full((batch,), 20.0),
                }

        batches = [
            (torch.zeros(2, 2, 3, 1, 1),
             torch.zeros(2, 2, 3, 1, 1),
             torch.tensor([[0.0, 20.0], [10.0, 30.0]])),
            (torch.zeros(1, 2, 3, 1, 1),
             torch.zeros(1, 2, 3, 1, 1),
             torch.tensor([[20.0, 40.0]])),
        ]
        model = ConstantModel()
        model.train()
        metrics, predictions = evaluate_driving_loader(model, batches)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(len(predictions["steering_true_deg"]), 3)
        self.assertNotIn("mae", metrics["steering_deg"])
        self.assertTrue(model.training)
        self.assertAlmostEqual(
            metrics["steering_deg"]["rmse"], (500.0 / 3.0) ** 0.5
        )

    def test_empty_loader(self):
        with self.assertRaises(ValueError):
            evaluate_driving_loader(nn.Identity(), [])


if __name__ == "__main__":
    unittest.main()
