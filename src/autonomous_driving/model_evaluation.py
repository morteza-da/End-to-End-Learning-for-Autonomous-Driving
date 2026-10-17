"""Evaluate a trained multimodal model on a labeled DBNet split."""
from __future__ import annotations

import math


def collect_driving_predictions(model, dataloader, *, device="cpu"):
    """Collect per-sequence ground truth and predictions without gradients.

    A DataLoader must yield (camera, lidar, targets), with targets [B, 2]
    in degrees and km/h. The model must output steering_deg and speed_kmh.
    """
    import torch

    was_training = model.training
    truth_steering, pred_steering = [], []
    truth_speed, pred_speed = [], []
    try:
        model.eval()
        with torch.no_grad():
            for camera, lidar, targets in dataloader:
                if targets.ndim != 2 or targets.shape[1] != 2:
                    raise ValueError("Expected targets shaped [B,2]")
                outputs = model(camera.to(device), lidar.to(device))
                steering = outputs["steering_deg"].detach().cpu().reshape(-1)
                speed = outputs["speed_kmh"].detach().cpu().reshape(-1)
                if steering.numel() != targets.shape[0] or speed.numel() != targets.shape[0]:
                    raise ValueError("Prediction/target batch size mismatch")
                y = targets.detach().cpu()
                truth_steering.extend(y[:, 0].tolist())
                pred_steering.extend(steering.tolist())
                truth_speed.extend(y[:, 1].tolist())
                pred_speed.extend(speed.tolist())
    finally:
        model.train(was_training)
    if not truth_steering:
        raise ValueError("Evaluation DataLoader is empty")
    if not all(math.isfinite(v) for group in
               (truth_steering, pred_steering, truth_speed, pred_speed)
               for v in group):
        raise ValueError("Nonfinite predictions or targets")
    return {
        "steering_true_deg": truth_steering,
        "steering_pred_deg": pred_steering,
        "speed_true_kmh": truth_speed,
        "speed_pred_kmh": pred_speed,
    }


def evaluate_driving_loader(model, dataloader, *, device="cpu"):
    """Calculate RMSE, R² and threshold accuracy over all observed samples."""
    from .experiment_reporting import evaluate_driving_predictions
    predictions = collect_driving_predictions(model, dataloader, device=device)
    metrics = evaluate_driving_predictions(
        predictions["steering_true_deg"], predictions["steering_pred_deg"],
        predictions["speed_true_kmh"], predictions["speed_pred_kmh"],
    )
    return metrics, predictions
