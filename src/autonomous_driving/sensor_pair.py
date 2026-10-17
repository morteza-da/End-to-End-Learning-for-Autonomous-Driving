"""Paired DBNet sensor loader for aligned camera and LiDAR CNN inputs."""
from __future__ import annotations

from pathlib import Path

from .dbnet_media import load_camera
from .lidar_projection import RangeImageConfig, load_lidar_rgb


def load_sensor_pair(camera_path, lidar_path, *, lidar_config=None):
    """Return float32 CHW camera and LiDAR RGB arrays plus LiDAR occupancy.

    Both branches are returned at the configured 299x299 default resolution.
    Targets are intentionally not inferred from raw CSV column positions.
    """
    config = lidar_config or RangeImageConfig()
    camera = load_camera(camera_path, size=(config.width, config.height))
    lidar, occupancy = load_lidar_rgb(lidar_path, config)
    if camera.shape != lidar.shape:
        raise ValueError(f"Sensor shape mismatch: {camera.shape} != {lidar.shape}")
    return {"camera": camera, "lidar": lidar, "lidar_occupancy": occupancy}


def load_indexed_pair(drive_path, frame_index, *, lidar_config=None):
    """Load an aligned numeric-stem JPEG/LAS pair from one DBNet drive."""
    drive = Path(drive_path)
    index = int(frame_index)
    if index < 0:
        raise ValueError("Frame index must be nonnegative")
    return load_sensor_pair(
        drive / "dvr_66x200" / f"{index}.jpg",
        drive / "points_16384" / f"{index}.las",
        lidar_config=lidar_config,
    )
