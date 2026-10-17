"""Check historical Git object paths for likely restricted files.

This is a filename/path heuristic, not a content/secret scan.
"""
from __future__ import annotations

import subprocess

from check_public_files import unsafe_tracked_path


def suspicious_historical_paths(raw: bytes) -> list[str]:
    """Parse NUL-separated `git log --name-only -z` output.

    The --format='' setting omits commit headers; blank records are ignored.
    """
    paths = {path.strip() for path in raw.decode("utf-8", errors="replace").split("\0")
             if path.strip()}
    return sorted(path for path in paths if unsafe_tracked_path(path))


def main():
    raw = subprocess.check_output([
        "git", "log", "--all", "--format=", "--name-only", "-z",
    ])
    suspicious = suspicious_historical_paths(raw)
    if suspicious:
        print("Potentially restricted filenames appear in reachable Git history:")
        for path in suspicious:
            print(f"  {path}")
        raise SystemExit(1)
    print("Reachable Git history filename check passed.")
    print("Not a source-content scan, secret scan, or proof of data licensing.")


if __name__ == "__main__":
    main()
