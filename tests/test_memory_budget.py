import unittest

from autonomous_driving.memory_budget import (
    check_temporal_memory, estimate_temporal_memory,
)


class MemoryBudgetTests(unittest.TestCase):
    def test_exact_single_layer_parameter_count(self):
        # LSTM: 4*H*I + 4*H*H + 8*H; attention H+1; head 2H+2
        estimate = estimate_temporal_memory(10, hidden_size=2)
        self.assertEqual(estimate.parameter_count, 80 + 16 + 16 + 3 + 6)
        self.assertAlmostEqual(
            estimate.adam_training_state_mib, 4 * estimate.parameter_mib
        )

    def test_multiple_layers_increase_memory(self):
        one = estimate_temporal_memory(128, hidden_size=8, num_layers=1)
        two = estimate_temporal_memory(128, hidden_size=8, num_layers=2)
        self.assertGreater(two.parameter_count, one.parameter_count)

    def test_guard_rejects_oversized_model(self):
        with self.assertRaises(MemoryError):
            check_temporal_memory(
                1_000_000, hidden_size=64, max_adam_state_mib=1
            )

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            estimate_temporal_memory(0)


if __name__ == "__main__":
    unittest.main()
