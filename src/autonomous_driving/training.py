"""Generic temporal-regression training helpers.

These train the temporal head on PRECOMPUTED fused features. End-to-end
backbone fine-tuning is a separate stage, not implemented by this helper.
"""
from __future__ import annotations


def joint_rmse_loss(outputs, targets, *, steering_weight: float = 1.0,
                    speed_weight: float = 1.0):
    """Weighted average of per-target RMSE; new reconstruction choice."""
    import torch
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Targets must have shape [batch,2]")
    if steering_weight <= 0 or speed_weight <= 0:
        raise ValueError("Weights must be positive")
    pred = torch.stack((outputs["steering_deg"], outputs["speed_kmh"]), dim=1)
    if pred.shape != targets.shape:
        raise ValueError("Prediction/target shape mismatch")
    per_target_rmse = torch.sqrt(torch.mean((pred - targets).square(), dim=0) + 1e-8)
    return (steering_weight * per_target_rmse[0] + speed_weight * per_target_rmse[1]) / (steering_weight + speed_weight)


def run_epoch(model, dataloader, optimizer=None, *, device="cpu",
              steering_weight: float = 1.0, speed_weight: float = 1.0):
    """Train when optimizer provided; otherwise run validation without gradients.

    DataLoader yields (fused_features [B,T,F], labels [B,2]).
    """
    import torch
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    sample_count = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(features)
            loss = joint_rmse_loss(outputs, labels,
                                   steering_weight=steering_weight,
                                   speed_weight=speed_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_count = labels.shape[0]
        weighted_loss += float(loss.detach()) * batch_count
        sample_count += batch_count
    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty DataLoader")
    return weighted_loss / sample_count
