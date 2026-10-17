import unittest
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
from autonomous_driving.training import joint_rmse_loss, run_epoch
from autonomous_driving.temporal_model import build_temporal_model


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class TrainingTests(unittest.TestCase):
    def test_zero_error_is_nearly_zero(self):
        out = {"steering_deg": torch.tensor([1., 2.]),
               "speed_kmh": torch.tensor([10., 20.])}
        targets = torch.tensor([[1., 10.], [2., 20.]])
        self.assertLess(float(joint_rmse_loss(out, targets)), 0.001)

    def test_training_and_validation_epoch(self):
        torch.manual_seed(7)
        model = build_temporal_model(5, hidden_size=4)
        loader = DataLoader(TensorDataset(torch.randn(6, 3, 5),
                                          torch.randn(6, 2)), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loss = run_epoch(model, loader, optimizer)
        val_loss = run_epoch(model, loader)
        self.assertTrue(torch.isfinite(torch.tensor([train_loss, val_loss])).all())


if __name__ == "__main__":
    unittest.main()
