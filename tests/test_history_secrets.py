"""Credential pattern scanner tests with synthetic strings only."""
import unittest

from scripts.check_history_secrets import detect_secret_patterns


class HistorySecretsTests(unittest.TestCase):
    def test_detects_private_key_header(self):
        self.assertIn(
            "private-key-header",
            detect_secret_patterns(b"-----BEGIN OPENSSH PRIVATE KEY-----"),
        )

    def test_detects_synthetic_aws_key(self):
        synthetic = b"AKIA" + b"A" * 16
        self.assertIn("aws-access-key", detect_secret_patterns(synthetic))

    def test_does_not_flag_normal_source(self):
        self.assertEqual(
            detect_secret_patterns(b"camera lidar fusion model weights and training"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
