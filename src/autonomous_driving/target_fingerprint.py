"""Stable fingerprint of canonical labels bound to sample and frame identity."""
from __future__ import annotations

import hashlib
import json
import math


def fingerprint_targets(observations):
    """Hash all target values, split/drive/frame identities; no sensor bytes."""
    records = []
    for row in observations:
        if not (math.isfinite(row.steering_deg) and math.isfinite(row.speed_kmh)):
            raise ValueError("Nonfinite target")
        records.append([
            row.split, row.drive_id, row.frame_index, row.sample_id,
            float(row.steering_deg).hex(), float(row.speed_kmh).hex(),
        ])
    if not records:
        raise ValueError("No labeled observations")
    records.sort()
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
