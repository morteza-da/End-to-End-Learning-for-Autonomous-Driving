"""Executable implementation of ImageNet-pretrained multimodal feature extraction.

Requires optional packages torch and timm. 
Pretrained weights may require a network connection and accepted model licenses.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backbone_config import get_backbone_assignment


@dataclass(frozen=True)
class BackboneModelNames:
    """Model registry names; override when a timm version uses other identifiers."""
    xception: str = "xception"
    resnext50: str = "resnext50_32x4d"

    def for_architecture(self, architecture: str) -> str:
        if architecture not in ("xception", "resnext50"):
            raise ValueError(f"Unsupported architecture: {architecture}")
        return getattr(self, architecture)


def available_model_names(pattern: str = "*xception*") -> list[str]:
    """Inspect installed timm registry rather than assuming a name exists."""
    import timm
    return timm.list_models(pattern)


def build_backbone(architecture: str, *, pretrained: bool = True,
                   names: BackboneModelNames | None = None):
    """Create a feature-map backbone (no classifier/global pooling)."""
    import timm
    names = names or BackboneModelNames()
    model_name = names.for_architecture(architecture)
    try:
        return timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not create {model_name!r}. Inspect timm.list_models(), "
            "package versions, internet access and pretrained weight availability."
        ) from exc


def freeze_all_except_last_n_children(model, n: int):
    """Example implementation freeze policy, not a previous research boundary.

    Freeze every parameter then unfreeze parameters in the last n direct child
    modules. Caller must inspect names and confirm nonempty trainable weights.
    """
    children = list(model.named_children())
    if not (1 <= n <= len(children)):
        raise ValueError("n must be within the model's direct-child count")
    for p in model.parameters():
        p.requires_grad = False
    for _, module in children[-n:]:
        for p in module.parameters():
            p.requires_grad = True
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters found; inspect model structure")
    return {
        "frozen_children": [name for name, _ in children[:-n]],
        "trainable_children": [name for name, _ in children[-n:]],
    }


def make_multimodal_backbones(configuration: str, *, pretrained: bool = True,
                              names: BackboneModelNames | None = None):
    """Return camera and LiDAR branches for documented configuration A or B."""
    cfg = get_backbone_assignment(configuration)
    return (
        build_backbone(cfg.camera_backbone, pretrained=pretrained, names=names),
        build_backbone(cfg.lidar_backbone, pretrained=pretrained, names=names),
    )


def flattened_concatenated_features(camera_model, lidar_model, camera, lidar):
    """Return [batch, concatenated_feature_dimension] for aligned frame pairs.

    Inputs are PyTorch tensors [B,3,299,299]. Input normalization is the
    caller's responsibility; do not assume raw RGB equals ImageNet-normalized.
    """
    import torch
    if camera.ndim != 4 or lidar.ndim != 4:
        raise ValueError("Expected two image tensors shaped [B, 3, 299, 299]")
    if camera.shape != lidar.shape or tuple(camera.shape[1:]) != (3, 299, 299):
        raise ValueError("Camera and LiDAR tensors must match [B,3,299,299]")
    camera_features = camera_model(camera)
    lidar_features = lidar_model(lidar)
    if camera_features.ndim != 4 or lidar_features.ndim != 4:
        raise ValueError("Backbones must output spatial feature maps [B,C,H,W]")
    return torch.cat(
        (torch.flatten(camera_features, start_dim=1),
         torch.flatten(lidar_features, start_dim=1)),
        dim=1,
    )
