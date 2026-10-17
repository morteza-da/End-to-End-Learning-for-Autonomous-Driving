"""Reject obvious raw-data, model-weight, and credential paths tracked by Git."""
from __future__ import annotations

from pathlib import PurePosixPath
import subprocess

FORBIDDEN_SUFFIXES = {".las", ".laz", ".jpg", ".jpeg", ".png", ".pt", ".pth", ".h5", ".ckpt", ".zip"}
FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json"}


def unsafe_tracked_path(path: str) -> bool:
    file = PurePosixPath(path)
    if file.name == ".gitkeep":
        return False
    if file.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    if file.name.lower() in FORBIDDEN_NAMES or file.name.startswith(".env."):
        return True
    return bool(file.parts and file.parts[0] in {"data", "datasets", "results", "checkpoints"})


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    unsafe = [path for path in paths if path and unsafe_tracked_path(path)]
    if unsafe:
        print("Potentially restricted files are tracked:")
        for path in unsafe:
            print(f"  {path}")
        raise SystemExit(1)
    print("Tracked-file policy passed (filename/path check only; no content or history scan).")


if __name__ == "__main__":
    main()
