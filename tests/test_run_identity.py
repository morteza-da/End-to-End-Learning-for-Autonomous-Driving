"""Tests for local Git revision capture."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from autonomous_driving.run_identity import git_run_identity


class RunIdentityTests(unittest.TestCase):
    def test_missing_repository_returns_explicit_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                git_run_identity(tmp),
                {"git_commit": None, "tracked_worktree_clean": None},
            )

    def test_detects_clean_and_modified_tracked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args], check=True,
                    capture_output=True, text=True,
                )
            git("init")
            git("config", "user.email", "test@example.org")
            git("config", "user.name", "CI Test")
            (root / "tracked.txt").write_text("initial", encoding="utf-8")
            git("add", "tracked.txt")
            git("commit", "-m", "initial")
            clean = git_run_identity(root)
            self.assertEqual(len(clean["git_commit"]), 40)
            self.assertTrue(clean["tracked_worktree_clean"])
            (root / "tracked.txt").write_text("changed", encoding="utf-8")
            dirty = git_run_identity(root)
            self.assertEqual(dirty["git_commit"], clean["git_commit"])
            self.assertFalse(dirty["tracked_worktree_clean"])


if __name__ == "__main__":
    unittest.main()
