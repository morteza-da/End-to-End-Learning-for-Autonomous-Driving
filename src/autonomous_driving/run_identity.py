"""Capture the Git revision used for an experiment without network access."""
from __future__ import annotations

from pathlib import Path
import subprocess


def git_run_identity(repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    try:
        commit = run("rev-parse", "--verify", "HEAD")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
            raise ValueError("Invalid Git revision")
        status = run("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"git_commit": None, "tracked_worktree_clean": None}
    return {"git_commit": commit, "tracked_worktree_clean": not bool(status)}
