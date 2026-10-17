"""Reproducible experiment manifest and evaluation artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
import random
from pathlib import Path

from .metrics import regression_metrics
from .threshold_metrics import driving_threshold_accuracies


def seed_everything(seed: int):
    """Seed Python, NumPy and PyTorch; report backend determinism separately."""
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def evaluate_driving_predictions(steering_true, steering_pred, speed_true, speed_pred):
    """Compute measured regression and threshold metrics for both outputs."""
    steering = regression_metrics(steering_true, steering_pred)
    speed = regression_metrics(speed_true, speed_pred)
    thresholds = driving_threshold_accuracies(
        steering_true, steering_pred, speed_true, speed_pred
    )
    return {
        "steering_deg": steering,
        "speed_kmh": speed,
        "threshold_accuracy_percent": thresholds,
        "sample_count": len(steering_true),
    }


def write_experiment_report(path, *, config, metrics, dataset_id,
                            split, checkpoint=None, git_commit=None,
                            notes=None):
    """Save explicitly supplied measurements; never synthesize metrics."""
    if not dataset_id or not split or metrics is None:
        raise ValueError("Dataset, split and measured metrics are required")
    import math

    def validate_metrics(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                validate_metrics(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                validate_metrics(value)
        elif obj is None:
            pass  # Explicitly undefined R² is serialized as JSON null.
        elif isinstance(obj, bool):
            raise ValueError("Boolean is not a metric")
        elif isinstance(obj, (int, float)):
            if not math.isfinite(obj):
                raise ValueError("Metrics must be finite")
        else:
            raise ValueError("Metrics must contain finite numbers or explicit null for undefined metrics")

    validate_metrics(metrics)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "split": split,
        "config": config.as_dict(),
        "metrics": metrics,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "git_commit": git_commit,
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "notes": notes,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return report
