"""Estimate the LSTM parameter footprint before allocating a temporal model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalMemoryEstimate:
    feature_width: int
    hidden_size: int
    num_layers: int
    parameter_count: int
    parameter_mib: float
    adam_training_state_mib: float


def estimate_temporal_memory(feature_width: int, hidden_size: int = 64,
                             num_layers: int = 1, bytes_per_element: int = 4
                             ) -> TemporalMemoryEstimate:
    """Estimate LSTM+attention+head parameter and Adam state memory.

    PyTorch LSTM: each layer has two 4H-wide weight matrices and two
    4H-wide biases. Adam training state estimates parameter, gradient,
    first moment and second moment (4 copies), excluding activations,
    backbone weights, temporary buffers and optimizer overhead.
    """
    if min(feature_width, hidden_size, num_layers, bytes_per_element) < 1:
        raise ValueError("Dimensions and element size must be positive")
    lstm = 0
    for layer in range(num_layers):
        input_width = feature_width if layer == 0 else hidden_size
        lstm += 4 * hidden_size * input_width
        lstm += 4 * hidden_size * hidden_size
        lstm += 8 * hidden_size
    attention = hidden_size + 1
    head = 2 * hidden_size + 2
    count = lstm + attention + head
    mib = count * bytes_per_element / (1024 ** 2)
    return TemporalMemoryEstimate(
        feature_width, hidden_size, num_layers, count, mib, 4 * mib
    )


def check_temporal_memory(feature_width: int, *, hidden_size: int = 64,
                          num_layers: int = 1, max_adam_state_mib: float = 1024):
    """Reject configurations exceeding a user-chosen LSTM-only memory cap."""
    if max_adam_state_mib <= 0:
        raise ValueError("Memory cap must be positive")
    estimate = estimate_temporal_memory(feature_width, hidden_size, num_layers)
    if estimate.adam_training_state_mib > max_adam_state_mib:
        raise MemoryError(
            f"Estimated temporal parameter+Adam state "
            f"{estimate.adam_training_state_mib:.1f} MiB exceeds "
            f"{max_adam_state_mib:.1f} MiB cap. This excludes CNNs and activations."
        )
    return estimate
