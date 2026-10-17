"""implementation configuration for research's two backbone assignments.

Source: research PDF pp. 105-106. Configuration here is declarative:
it defines model configurations without loading pretrained weights.
"""
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
            raise ValueError("research input resolution is 299 x 299")


CONFIGURATIONS = {
    "A": BackboneAssignment("A", camera_backbone="resnext50", lidar_backbone="xception"),
    "B": BackboneAssignment("B", camera_backbone="xception", lidar_backbone="resnext50"),
}


def get_backbone_assignment(name: str) -> BackboneAssignment:
    """Return immutable modality-to-backbone assignment from the research."""
    try:
        return CONFIGURATIONS[name.upper()]
    except (KeyError, AttributeError) as exc:
        raise ValueError("Expected configuration A or B") from exc


@dataclass(frozen=True)
class FineTuningPlan:
    """Explicit *new experiment* plan; previous frozen blocks are unknown.

    Frozen/trainable module names must be supplied after inspecting a concrete
    implementation. No arbitrary previous freeze boundary is assumed.
    """

    frozen_module_names: tuple[str, ...]
    trainable_module_names: tuple[str, ...]

    def __post_init__(self) -> None:
        frozen = set(self.frozen_module_names)
        trainable = set(self.trainable_module_names)
        if not frozen or not trainable:
            raise ValueError("Both frozen and trainable module sets must be nonempty")
        if frozen & trainable:
            raise ValueError("Frozen and trainable module names must not overlap")
