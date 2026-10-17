"""Camera–LiDAR backbone assignments and fine-tuning configuration."""
from __future__ import annotations

from dataclasses import dataclass

IMAGE_SIZE = (299, 299)
SUPPORTED_BACKBONES = frozenset({"xception", "resnext50"})


@dataclass(frozen=True)
class BackboneAssignment:
    name: str
    camera_backbone: str
    lidar_backbone: str
    image_size: tuple[int, int] = IMAGE_SIZE

    def __post_init__(self) -> None:
        if self.name not in {"A", "B"}:
            raise ValueError("Configuration must be A or B")
        if {self.camera_backbone, self.lidar_backbone} != SUPPORTED_BACKBONES:
            raise ValueError("Each configuration must use one Xception and one ResNeXt-50")
        if self.image_size != IMAGE_SIZE:
            raise ValueError("Model input resolution is 299 x 299")


CONFIGURATIONS = {
    "A": BackboneAssignment("A", camera_backbone="resnext50", lidar_backbone="xception"),
    "B": BackboneAssignment("B", camera_backbone="xception", lidar_backbone="resnext50"),
}


def get_backbone_assignment(name: str) -> BackboneAssignment:
    """Return immutable modality-to-backbone assignment for multimodal experiments."""
    try:
        return CONFIGURATIONS[name.upper()]
    except (KeyError, AttributeError) as exc:
        raise ValueError("Expected configuration A or B") from exc


@dataclass(frozen=True)
class FineTuningPlan:
    """Specify frozen and trainable modules for a concrete backbone.\n\n    Inspect the model registry and record the chosen module boundaries.\n    """

    frozen_module_names: tuple[str, ...]
    trainable_module_names: tuple[str, ...]

    def __post_init__(self) -> None:
        frozen = set(self.frozen_module_names)
        trainable = set(self.trainable_module_names)
        if not frozen or not trainable:
            raise ValueError("Both frozen and trainable module sets must be nonempty")
        if frozen & trainable:
            raise ValueError("Frozen and trainable module names must not overlap")
