"""Optional spatial pooling before concatenating CNN feature maps.

This is an explicit architectural alternative to full spatial flattening.
"""
from __future__ import annotations


def compressed_feature_width(camera_shape, lidar_shape, *, pool_size=1):
    """Compute fused width from two [B,C,H,W] output shapes."""
    if not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    shapes = (tuple(camera_shape), tuple(lidar_shape))
    if any(len(shape) != 4 or min(shape) < 1 for shape in shapes):
        raise ValueError("Expected positive [B,C,H,W] shapes")
    if shapes[0][0] != shapes[1][0]:
        raise ValueError("Backbone batch sizes must match")
    return (shapes[0][1] + shapes[1][1]) * pool_size * pool_size


def pool_and_concatenate(camera_features, lidar_features, *, pool_size=1):
    """Adaptive-average-pool both spatial tensors; preserve gradients."""
    import torch
    from torch.nn import functional as F

    if camera_features.ndim != 4 or lidar_features.ndim != 4:
        raise ValueError("Expected spatial [B,C,H,W] tensors")
    if camera_features.shape[0] != lidar_features.shape[0]:
        raise ValueError("Backbone batch sizes must match")
    if not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    camera = F.adaptive_avg_pool2d(camera_features, (pool_size, pool_size))
    lidar = F.adaptive_avg_pool2d(lidar_features, (pool_size, pool_size))
    return torch.cat((camera.flatten(1), lidar.flatten(1)), dim=1)
