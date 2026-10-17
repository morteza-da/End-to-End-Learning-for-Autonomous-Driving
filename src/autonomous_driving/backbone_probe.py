"""Inspect real backbone feature-map dimensions without assuming output widths."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .backbone_config import get_backbone_assignment
from .backbone_extraction import BackboneModelNames, build_backbone
from .memory_budget import estimate_temporal_memory


@dataclass(frozen=True)
class BackboneProbe:
    assignment: str
    camera_architecture: str
    lidar_architecture: str
    camera_shape: tuple[int, ...]
    lidar_shape: tuple[int, ...]
    fused_feature_width: int
    temporal_parameters: int
    temporal_adam_state_mib: float

    def as_dict(self):
        return asdict(self)


def probe_backbones(assignment, *, pretrained=False, names=None,
                    hidden_size=64, num_layers=1, device="cpu"):
    """Run one synthetic image per branch; no real dataset or training.

    Report parameter-state estimates only, not total training RAM/VRAM.
    """
    import torch
    cfg = get_backbone_assignment(assignment)
    names = names or BackboneModelNames()
    camera = build_backbone(cfg.camera_backbone, pretrained=pretrained, names=names)
    lidar = build_backbone(cfg.lidar_backbone, pretrained=pretrained, names=names)
    try:
        camera.to(device).eval()
        lidar.to(device).eval()
        with torch.inference_mode():
            sample = torch.zeros(1, 3, 299, 299, device=device)
            camera_features = camera(sample)
            lidar_features = lidar(sample)
        if camera_features.ndim != 4 or lidar_features.ndim != 4:
            raise ValueError("Backbones must output spatial [B,C,H,W] tensors")
        if camera_features.shape[0] != 1 or lidar_features.shape[0] != 1:
            raise ValueError("Unexpected backbone batch dimension")
        camera_width = camera_features.numel()
        lidar_width = lidar_features.numel()
        width = camera_width + lidar_width
        estimate = estimate_temporal_memory(
            width, hidden_size=hidden_size, num_layers=num_layers
        )
        return BackboneProbe(
            assignment=assignment,
            camera_architecture=cfg.camera_backbone,
            lidar_architecture=cfg.lidar_backbone,
            camera_shape=tuple(camera_features.shape),
            lidar_shape=tuple(lidar_features.shape),
            fused_feature_width=width,
            temporal_parameters=estimate.parameter_count,
            temporal_adam_state_mib=estimate.adam_training_state_mib,
        )
    finally:
        del camera, lidar
