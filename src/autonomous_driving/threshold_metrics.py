"""Dissertation-defined threshold accuracy for continuous driving outputs.

Thresholds from dissertation PDF p. 110:
steering angle: absolute error <= 6 degrees
vehicle speed: absolute error <= 5 km/h
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def threshold_accuracy(
    y_true: Sequence[float], y_pred: Sequence[float], tolerance: float
) -> float:
    """Return percentage of predictions within inclusive absolute-error tolerance.

    Raises ValueError for empty/mismatched sequences, invalid thresholds, or
    non-finite labels/predictions. Inputs must use the same physical units.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be a finite nonnegative number")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Expected nonempty equal-length targets and predictions")
    pairs = [(float(a), float(b)) for a, b in zip(y_true, y_pred)]
    if not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
        raise ValueError("Targets and predictions must be finite")
    accepted = sum(abs(a - b) <= tolerance for a, b in pairs)
    return 100.0 * accepted / len(pairs)


def driving_threshold_accuracies(
    steering_true_deg: Sequence[float],
    steering_pred_deg: Sequence[float],
    speed_true_kmh: Sequence[float],
    speed_pred_kmh: Sequence[float],
) -> dict[str, float]:
    """Compute separate dissertation threshold accuracies for both targets.

    The dissertation reports per-target percentages; no joint sample-level
    accuracy is implied by this function.
    """
    lengths = {
        len(steering_true_deg), len(steering_pred_deg),
        len(speed_true_kmh), len(speed_pred_kmh)
    }
    if len(lengths) != 1:
        raise ValueError("All steering and speed sequences must have equal length")
    return {
        "steering_accuracy_percent": threshold_accuracy(
            steering_true_deg, steering_pred_deg, tolerance=6.0
        ),
        "speed_accuracy_percent": threshold_accuracy(
            speed_true_kmh, speed_pred_kmh, tolerance=5.0
        ),
    }
