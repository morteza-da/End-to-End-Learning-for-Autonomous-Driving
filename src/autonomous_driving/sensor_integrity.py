"""Content SHA-256 for local camera and LiDAR files referenced by observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint_sensor_files(root, observations):
    """Read each unique sensor file once; never copy or upload source bytes."""
    root = Path(root).resolve()
    paths = sorted({p for row in observations for p in (row.camera_path, row.lidar_path)})
    if not paths:
        raise ValueError("No sensor files to verify")
    entries = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Sensor path escapes dataset root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append((relative, path.stat().st_size, digest.hexdigest()))
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
    return {
        "file_count": len(entries),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "algorithm": "sha256-per-file-then-manifest",
    }
