"""Joint camera-LiDAR backbone and temporal regression training interface."""
from __future__ import annotations


def build_end_to_end_model(camera_backbone, lidar_backbone, *,
                           input_size: int, hidden_size: int = 64,
                           num_layers: int = 1):
    """Combine two spatial CNN branches, Flatten/Concat, LSTM and attention.

    input_size must equal the actual concatenated flattened feature width.
    """
    import torch
    from torch import nn
    from .temporal_model import build_temporal_model, stack_temporal_features

    class EndToEndDrivingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_backbone = camera_backbone
            self.lidar_backbone = lidar_backbone
            self.temporal = build_temporal_model(
                input_size, hidden_size=hidden_size, num_layers=num_layers
            )

        def forward(self, camera_frames, lidar_frames):
            fused = stack_temporal_features(
                self.camera_backbone, self.lidar_backbone,
                camera_frames, lidar_frames
            )
            return self.temporal(fused)

    return EndToEndDrivingModel()


def infer_feature_width(camera_backbone, lidar_backbone, *, device="cpu"):
    """Infer flattened feature width from one 299x299 pair without gradients."""
    import torch
    from .backbone_extraction import flattened_concatenated_features
    was_camera_training = camera_backbone.training
    was_lidar_training = lidar_backbone.training
    try:
        camera_backbone.eval()
        lidar_backbone.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 3, 299, 299, device=device)
            features = flattened_concatenated_features(
                camera_backbone, lidar_backbone, sample, sample
            )
        return int(features.shape[1])
    finally:
        camera_backbone.train(was_camera_training)
        lidar_backbone.train(was_lidar_training)


def run_multimodal_epoch(model, dataloader, optimizer=None, *, device="cpu",
                         steering_weight=1.0, speed_weight=1.0):
    """Train or evaluate on (camera[B,T,3,299,299],
    lidar[B,T,3,299,299], targets[B,2]) batches.

    Reported loss is sample-weighted average of batch-level RMSE objectives,
    not a dataset-global RMSE. Use independent aggregate metrics for reporting.
    """
    import torch
    from .training import joint_rmse_loss
    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    for camera, lidar, targets in dataloader:
        camera = camera.to(device)
        lidar = lidar.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(camera, lidar)
            loss = joint_rmse_loss(
                outputs, targets,
                steering_weight=steering_weight, speed_weight=speed_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = int(targets.shape[0])
        total += float(loss.detach()) * n
        samples += n
    if samples == 0:
        raise ValueError("Empty multimodal DataLoader")
    return total / samples
