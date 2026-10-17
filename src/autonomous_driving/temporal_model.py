"""LSTM -> additive temporal attention -> joint steering/speed regression.

A new implementation choice, NOT a verified previous attention mechanism.
Requires PyTorch only when instantiating the model.
"""
from __future__ import annotations


def build_temporal_model(input_size: int, hidden_size: int = 64,
                         num_layers: int = 1):
    """Create a batch-first model accepting [batch,time,fused_features].

    Input features are fed DIRECTLY into LSTM, without a pre-LSTM Dense layer.
    """
    if input_size < 1 or hidden_size < 1 or num_layers < 1:
        raise ValueError("Model dimensions and layer count must be positive")
    import torch
    from torch import nn

    class TemporalDrivingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True)
            self.attention_score = nn.Linear(hidden_size, 1)
            self.regression = nn.Linear(hidden_size, 2)

        def forward(self, fused_sequence):
            if fused_sequence.ndim != 3 or fused_sequence.shape[-1] != input_size:
                raise ValueError("Expected [batch, sequence_length, input_size]")
            if fused_sequence.shape[1] < 2:
                raise ValueError("Expected multiple consecutive frames")
            sequence_features, _ = self.lstm(fused_sequence)
            weights = torch.softmax(self.attention_score(sequence_features), dim=1)
            context = (weights * sequence_features).sum(dim=1)
            predictions = self.regression(context)
            return {
                "steering_deg": predictions[:, 0],
                "speed_kmh": predictions[:, 1],
                "attention_weights": weights.squeeze(-1),
            }

    return TemporalDrivingModel()


def stack_temporal_features(camera_model, lidar_model, camera_frames, lidar_frames):
    """Extract per-frame features and stack them as [batch,time,features].

    Inputs [B,T,3,299,299]. Uses the exact Flatten -> Concatenate function.
    """
    from .backbone_extraction import flattened_concatenated_features
    if camera_frames.ndim != 5 or lidar_frames.shape != camera_frames.shape:
        raise ValueError("Both modalities must be aligned [B,T,3,299,299]")
    if tuple(camera_frames.shape[2:]) != (3, 299, 299) or camera_frames.shape[1] < 2:
        raise ValueError("Expected >=2 frames per sequence, each [3,299,299]")
    import torch
    b, t = camera_frames.shape[:2]
    cam = camera_frames.reshape(b * t, 3, 299, 299)
    lidar = lidar_frames.reshape(b * t, 3, 299, 299)
    fused = flattened_concatenated_features(camera_model, lidar_model, cam, lidar)
    return fused.reshape(b, t, -1)
