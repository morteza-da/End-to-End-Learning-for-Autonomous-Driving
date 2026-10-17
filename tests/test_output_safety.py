"""Prevent experiment output from modifying local raw dataset trees."""
from pathlib import Path
import tempfile
import unittest

from autonomous_driving.output_safety import (
    require_output_outside_dataset, require_fresh_experiment_directory
)


class OutputSafetyTests(unittest.TestCase):
    def test_rejects_dataset_root_and_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dbnet"
            root.mkdir()
            for output in (root, root / "results", root / "train" / "outputs"):
                with self.subTest(output=str(output)):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        require_output_outside_dataset(root, output)

    def test_accepts_sibling_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dbnet"
            root.mkdir()
            output = Path(tmp) / "results"
            self.assertEqual(require_output_outside_dataset(root, output), output.resolve())

    def test_existing_experiment_artifacts_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "A" / "pool-2x2"
            output.mkdir(parents=True)
            (output / "best.pt").write_bytes(b"prior-checkpoint")
            with self.assertRaisesRegex(ValueError, "not empty"):
                require_fresh_experiment_directory(output)
            self.assertEqual(
                require_fresh_experiment_directory(output, allow_existing=True),
                output.resolve()
            )
            self.assertEqual((output / "best.pt").read_bytes(), b"prior-checkpoint")

    def test_new_experiment_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new-run"
            self.assertEqual(require_fresh_experiment_directory(output), output.resolve())
            self.assertFalse(output.exists())

    def test_resolves_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dbnet"
            root.mkdir()
            output = root / ".." / "dbnet" / "outputs"
            with self.assertRaises(ValueError):
                require_output_outside_dataset(root, output)


if __name__ == "__main__":
    unittest.main()
