"""Unit tests for the reachable Git history filename audit."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_git_history import suspicious_historical_paths


class GitHistoryPolicyTests(unittest.TestCase):
    def test_detects_restricted_historical_paths(self):
        raw = b"README.md\0data/train/0.jpg\0src/model.py\0.env.local\0"
        self.assertEqual(
            suspicious_historical_paths(raw),
            [".env.local", "data/train/0.jpg"],
        )

    def test_ignores_source_and_gitkeep(self):
        raw = b"README.md\0data/.gitkeep\0docs/guide.md\0"
        self.assertEqual(suspicious_historical_paths(raw), [])


if __name__ == "__main__":
    unittest.main()
