"""Record lightweight runtime and dependency metadata for reproducibility."""
from __future__ import annotations

from importlib import metadata
import platform
import sys


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment_identity() -> dict:
    """Collect installed package versions without importing GPU libraries."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "dependencies": {
            name: installed_version(name)
            for name in ("torch", "torchvision", "timm", "numpy", "Pillow")
        },
    }
