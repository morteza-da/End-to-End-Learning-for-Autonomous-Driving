"""Serializable configuration for reproducible multimodal experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    assignment: str = "A"
    image_size: int = 299
    sequence_length: int = 4
    stride: int = 1
    batch_size: int = 1
    hidden_size: int = 64
    lstm_layers: int = 1
    seed: int = 42
    learning_rate: float = 1e-4
    epochs: int = 10
    pretrained: bool = True
    pool_size: int | None = None
    camera_trainable_last_n: int = 1
    lidar_trainable_last_n: int = 1
    steering_weight: float = 1.0
    speed_weight: float = 1.0

    def __post_init__(self):
        from .backbone_config import get_backbone_assignment
        get_backbone_assignment(self.assignment)
        if self.image_size != 299:
            raise ValueError("Current backbone interface requires 299x299 images")
        if min(self.sequence_length, self.stride, self.batch_size,
               self.hidden_size, self.lstm_layers, self.epochs,
               self.camera_trainable_last_n, self.lidar_trainable_last_n) < 1:
            raise ValueError("Count and dimension settings must be positive")
        if self.sequence_length < 2:
            raise ValueError("Temporal windows must contain at least two frames")
        if self.pool_size is not None and (type(self.pool_size) is not int or self.pool_size not in (1, 2, 5)):
            raise ValueError("pool_size must be None, 1, 2, or 5")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if self.learning_rate <= 0 or self.steering_weight <= 0 or self.speed_weight <= 0:
            raise ValueError("Learning rate and loss weights must be positive")

    def as_dict(self):
        return asdict(self)


def save_experiment_config(config: ExperimentConfig, path):
    """Write config atomically to an output path outside the raw dataset."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(destination)


def load_experiment_config(path):
    """Reject unknown keys instead of silently ignoring experiment settings."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Experiment config must be a JSON object")
    return ExperimentConfig(**data)
