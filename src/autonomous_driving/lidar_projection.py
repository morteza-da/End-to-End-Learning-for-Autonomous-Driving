"""Spherical LiDAR range-image projection for RGB CNN inputs.

Azimuth/elevation rasterization is a configurable processing method. It does
not require camera calibration and is not a camera-view projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RangeImageConfig:
    width: int = 299
    height: int = 299
    azimuth_min_deg: float = -180.0
    azimuth_max_deg: float = 180.0
    elevation_min_deg: float = -25.0
    elevation_max_deg: float = 15.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not (-360 <= self.azimuth_min_deg < self.azimuth_max_deg <= 360):
            raise ValueError("Invalid azimuth limits")
        if not (-90 <= self.elevation_min_deg < self.elevation_max_deg <= 90):
            raise ValueError("Invalid elevation limits")


def project_xyz_to_rgb(xyz, config: RangeImageConfig = RangeImageConfig()):
    """Convert Nx3 XYZ to a uint8 HxWx3 distance-coded range image.

    Points are projected by azimuth and elevation. When multiple points fall
    into one pixel, the nearest 3D return is retained. Empty pixels are black.
    Returns (image, occupied_mask); coordinates are in the LiDAR sensor frame.
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    if not np.isfinite(points).all():
        raise ValueError("LiDAR points must be finite")

    image = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    occupied = np.zeros((config.height, config.width), dtype=bool)
    if not len(points):
        return image, occupied

    x, y, z = points.T
    radial_xy = np.hypot(x, y)
    radial_3d = np.hypot(radial_xy, z)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, radial_xy))
    valid = ((radial_3d > 0) &
             (azimuth >= config.azimuth_min_deg) &
             (azimuth <= config.azimuth_max_deg) &
             (elevation >= config.elevation_min_deg) &
             (elevation <= config.elevation_max_deg))
    if not np.any(valid):
        return image, occupied

    d = radial_xy[valid]
    distance_3d = radial_3d[valid]
    u = ((azimuth[valid] - config.azimuth_min_deg) /
         (config.azimuth_max_deg - config.azimuth_min_deg))
    v = ((config.elevation_max_deg - elevation[valid]) /
         (config.elevation_max_deg - config.elevation_min_deg))
    col = np.minimum((u * config.width).astype(np.int64), config.width - 1)
    row = np.minimum((v * config.height).astype(np.int64), config.height - 1)
    col = np.maximum(col, 0)
    row = np.maximum(row, 0)

    # Stable nearest-first order: the first return for each pixel is retained.
    order = np.argsort(distance_3d, kind="stable")
    pixel_ids = row[order] * config.width + col[order]
    _, first = np.unique(pixel_ids, return_index=True)
    selected = order[first]
    row, col, d = row[selected], col[selected], d[selected]

    blue = d < 20
    green = (d >= 20) & (d < 40)
    red = d >= 40
    image[row[blue], col[blue], 2] = np.clip(np.rint(255 * d[blue] / 20), 0, 255).astype(np.uint8)
    image[row[green], col[green], 1] = np.clip(np.rint(255 * d[green] / 40), 0, 255).astype(np.uint8)
    image[row[red], col[red], 0] = np.clip(np.rint(255 * d[red] / 60), 0, 255).astype(np.uint8)
    occupied[row, col] = True
    return image, occupied


def load_lidar_rgb(path, config: RangeImageConfig = RangeImageConfig()):
    """Decode a LAS file and return float32 CHW RGB plus occupancy mask."""
    from .dbnet_media import load_lidar_xyz
    image, mask = project_xyz_to_rgb(load_lidar_xyz(path), config)
    return image.astype(np.float32).transpose(2, 0, 1) / 255.0, mask
