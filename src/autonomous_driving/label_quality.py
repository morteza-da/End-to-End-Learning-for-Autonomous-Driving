"""Read-only quality checks for canonical steering and speed labels."""
from __future__ import annotations

import math
from collections import defaultdict


def audit_target_ranges(observations, *, steering_limit_deg=40.0, speed_limit_kmh=120.0):
    """Summarize measured labels and flag, never silently clamp, outliers."""
    rows = list(observations)
    if not rows:
        raise ValueError("No labeled observations")
    if not (math.isfinite(steering_limit_deg) and steering_limit_deg > 0
            and math.isfinite(speed_limit_kmh) and speed_limit_kmh > 0):
        raise ValueError("Limits must be positive finite numbers")
    grouped = defaultdict(list)
    for row in rows:
        if not (math.isfinite(row.steering_deg) and math.isfinite(row.speed_kmh)):
            raise ValueError("Nonfinite target")
        grouped[row.split].append(row)
    summary = {}
    for split, group in sorted(grouped.items()):
        angles = [r.steering_deg for r in group]
        speeds = [r.speed_kmh for r in group]
        summary[split] = {
            "sample_count": len(group),
            "steering_deg": {"min": min(angles), "max": max(angles)},
            "speed_kmh": {"min": min(speeds), "max": max(speeds)},
            "steering_out_of_range": sum(abs(x) > steering_limit_deg for x in angles),
            "speed_out_of_range": sum(x < 0 or x > speed_limit_kmh for x in speeds),
        }
    return summary


def require_plausible_targets(summary):
    for split, result in summary.items():
        if result["steering_out_of_range"] or result["speed_out_of_range"]:
            raise ValueError(f"{split}: target labels outside configured physical limits")
