"""Explicit, validated mapping from raw DBNet label channels to model targets.

Channel order and units are configuration inputs, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .sequence_manifest import Observation, validate_observations


@dataclass(frozen=True)
class TargetSchema:
    steering_channel: int
    speed_channel: int
    steering_unit: str
    speed_unit: str

    def __post_init__(self):
        if self.steering_channel < 0 or self.speed_channel < 0:
            raise ValueError("Channel indices must be nonnegative")
        if self.steering_channel == self.speed_channel:
            raise ValueError("Steering and speed must use different channels")
        if self.steering_unit not in {"deg", "rad"}:
            raise ValueError("Steering unit must be 'deg' or 'rad'")
        if self.speed_unit not in {"km/h", "m/s", "mph"}:
            raise ValueError("Speed unit must be 'km/h', 'm/s', or 'mph'")


def map_raw_observations(records, schema: TargetSchema):
    """Build verified-unit Observation objects from raw DBNet records.

    Records must come from build_labeled_observations. The schema must be
    independently confirmed using the dataset's authoritative metadata.
    """
    mapped = []
    for record in records:
        channels = record["raw_label_columns"]
        if max(schema.steering_channel, schema.speed_channel) >= len(channels):
            raise ValueError(f"Insufficient label channels: {record['sample_id']}")
        steering = float(channels[schema.steering_channel])
        speed = float(channels[schema.speed_channel])
        if not math.isfinite(steering) or not math.isfinite(speed):
            raise ValueError(f"Nonfinite label: {record['sample_id']}")
        if schema.steering_unit == "rad":
            steering = math.degrees(steering)
        if schema.speed_unit == "m/s":
            speed *= 3.6
        elif schema.speed_unit == "mph":
            speed *= 1.609344
        mapped.append(Observation(
            sample_id=record["sample_id"],
            drive_id=record["drive_id"],
            frame_index=record["frame_index"],
            split=record["split"],
            camera_path=record["camera_path"],
            lidar_path=record["lidar_path"],
            steering_deg=steering,
            speed_kmh=speed,
        ))
    return validate_observations(mapped)
