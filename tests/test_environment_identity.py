"""Tests for non-invasive runtime metadata collection."""
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from autonomous_driving.environment_identity import environment_identity, installed_version


class EnvironmentIdentityTests(unittest.TestCase):
    def test_missing_package_is_recorded_as_unknown(self):
        with patch(
            "autonomous_driving.environment_identity.metadata.version",
            side_effect=PackageNotFoundError,
        ):
            self.assertIsNone(installed_version("missing-package"))

    def test_records_python_and_expected_dependency_names(self):
        result = environment_identity()
        self.assertTrue(result["python"])
        self.assertTrue(result["python_implementation"])
        self.assertTrue(result["platform"])
        self.assertTrue(result["python_executable"])
        self.assertEqual(
            set(result["dependencies"]),
            {"torch", "torchvision", "timm", "numpy", "Pillow"},
        )


if __name__ == "__main__":
    unittest.main()
