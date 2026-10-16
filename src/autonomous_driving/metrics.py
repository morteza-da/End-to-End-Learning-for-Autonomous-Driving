"""Framework-independent regression metrics for steering and speed.

All values are computed from paired ground-truth and prediction arrays.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Return MAE, RMSE and R²; reject empty or mismatched inputs.

    R² is undefined when the target has zero total variance.
    """
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected nonempty equal-length ground truth and predictions")
    truth = [float(v) for v in y_true]
    pred = [float(v) for v in y_pred]
    if not all(math.isfinite(v) for v in truth + pred):
        raise ValueError("Inputs must be finite numbers")
    errors = [a - b for a, b in zip(truth, pred)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean = sum(truth) / len(truth)
    total = sum((v - mean) ** 2 for v in truth)
    residual = sum(e * e for e in errors)
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}
