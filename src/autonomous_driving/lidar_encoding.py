"""Dissertation-guided LiDAR distance and RGB color encoding.

Source: supplied doctoral dissertation, PDF p. 73, Equation (3-1).
This module deliberately does NOT rasterize/project 3D points to 2D pixels:
the original pixel mapping, calibration, and collision policy are unknown.

The piecewise branches below are a transparent interpretation of the printed
equation, not a claim of byte-for-byte reproduction of lost original code.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def planar_distance(px: float, py: float) -> float:
    """Return sqrt(px**2 + py**2) as specified in the dissertation.

    Pz is intentionally absent from the source equation.
    """
    if not math.isfinite(px) or not math.isfinite(py):
        raise ValueError("LiDAR coordinates must be finite")
    return math.hypot(px, py)


def _uint8(value: float) -> int:
    """Round to nearest integer and clamp to the uint8 RGB range."""
    return max(0, min(255, int(round(value))))


def distance_to_rgb(distance: float) -> tuple[int, int, int]:
    """Return (red, green, blue) for one planar LiDAR distance.

    Interpretation of dissertation Eq. (3-1):
      distance < 20:  B = 255 * distance / 20
      20 <= distance < 40: G = 255 * distance / 40
      distance >= 40: R = min(255, 255 * distance / 60)

    Inactive channels are zero. Integer rounding is a reconstruction choice.
    """
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Distance must be finite and nonnegative")
    if distance < 20.0:
        return (0, 0, _uint8(255.0 * distance / 20.0))
    if distance < 40.0:
        return (0, _uint8(255.0 * distance / 40.0), 0)
    return (_uint8(min(255.0, 255.0 * distance / 60.0)), 0, 0)


def encode_points_xy(points: Iterable[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Encode point coordinates into RGB colors; does not create an image."""
    return [distance_to_rgb(planar_distance(px, py)) for px, py in points]
