"""Unit tests for README link verification."""
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_readme_links import broken_local_links


class ReadmeLinkTests(unittest.TestCase):
    def test_local_missing_and_external_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs" / "ok.md").write_text("ok", encoding="utf-8")
            markdown = (
                "[working](docs/ok.md) [anchor](#overview) "
                "[external](https://example.org/page) "
                "[missing](docs/missing.md) [escape](../outside.md)"
            )
            self.assertEqual(
                broken_local_links(markdown, root),
                ["../outside.md", "docs/missing.md"],
            )

    def test_url_encoded_local_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "some file.md").write_text("ok", encoding="utf-8")
            self.assertEqual(broken_local_links("[file](some%20file.md)", root), [])


if __name__ == "__main__":
    unittest.main()
