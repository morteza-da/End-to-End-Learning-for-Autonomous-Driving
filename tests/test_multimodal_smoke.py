"""Synthetic smoke-test checks (PyTorch optional in base environment)."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch optional dependency not installed")
class MultimodalSmokeTests(unittest.TestCase):
    def test_forward_evaluation_contract(self):
        import importlib.util\n        from pathlib import Path\n        path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_multimodal.py"\n        spec = importlib.util.spec_from_file_location("smoke_test_multimodal", path)\n        module = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(module)\n        run_smoke_test = module.run_smoke_test
        result = run_smoke_test(batch_size=2, sequence_length=2)
        self.assertEqual(result["data_type"], "synthetic")
        self.assertFalse(result["performance_claim"])
        self.assertEqual(result["feature_width"], 4)
        self.assertEqual(result["attention_shape"], [2, 2])
        self.assertEqual(result["prediction_count"], 2)

    def test_reject_short_sequences(self):
        from scripts.smoke_test_multimodal import run_smoke_test
        with self.assertRaises(ValueError):
            run_smoke_test(sequence_length=1)


if __name__ == "__main__":
    unittest.main()
