"""Stable fingerprints of frame-index-based temporal window membership.

Hashes record manifest identity, not physical sensor timestamp synchronization.
"""
from __future__ import annotations

import hashlib
import json


def fingerprint_windows(windows):
    """Hash sorted split/drive/frame/sample/path membership without raw data bytes."""
    records = []
    for window in windows:
        if not window:
            raise ValueError("Empty temporal window")
        records.append([
            {
                "split": observation.split,
                "drive_id": observation.drive_id,
                "frame_index": observation.frame_index,
                "sample_id": observation.sample_id,
                "camera_path": observation.camera_path,
                "lidar_path": observation.lidar_path,
            }
            for observation in window
        ])
    if not records:
        raise ValueError("No temporal windows to fingerprint")
    records.sort(key=lambda window: json.dumps(window, sort_keys=True))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
