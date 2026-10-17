"""Execute the DBNet preflight CLI against a temporary synthetic dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_dbnet_preflight.py"


class PreflightCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "dbnet"
        for split in ("train", "val", "test"):
            drive = self.data / split / "1"
            camera = drive / "dvr_66x200"
            lidar = drive / "points_16384"
            camera.mkdir(parents=True)
            lidar.mkdir(parents=True)
            for i in range(5):
                (camera / f"{i}.jpg").write_bytes(b"jpg" + bytes([i]))
                (lidar / f"{i}.las").write_bytes(b"las" + bytes([i]))
            if split != "test":
                (drive / "behavior.csv").write_text(
                    "".join(f"{i},{20+i}\n" for i in range(5)), encoding="utf-8"
                )
        self.evidence = self.root / "evidence.json"
        self.evidence.write_text(json.dumps({
            "source_title": "Synthetic test fixture (not real dataset evidence)",
            "source_url": "https://example.org/synthetic-test-only",
            "evidence_excerpt": "Synthetic test fixture, not scientific evidence.",
            "reviewed_by": "Automated test",
            "review_date": "2026-10-17",
            "steering_channel": 0,
            "speed_channel": 1,
            "steering_unit": "deg",
            "speed_unit": "km/h",
        }), encoding="utf-8")
        self.output = self.root / "report.json"

    def run_cli(self, *extra):
        env = os.environ.copy()
        source = str(SCRIPT.parent.parent / "src")
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run([
            sys.executable, str(SCRIPT),
            "--data-root", str(self.data),
            "--label-provenance", str(self.evidence),
            "--steering-channel", "0", "--speed-channel", "1",
            "--steering-unit", "deg", "--speed-unit", "km/h",
            "--sequence-length", "4", "--output", str(self.output),
            *extra,
        ], capture_output=True, text=True, env=env, check=False)

    def test_cli_passes_and_writes_hashed_report(self):
        result = self.run_cli("--hash-sensor-files")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(report["readiness"]["passed"])
        self.assertEqual(report["window_counts"]["train"], 2)
        self.assertEqual(report["window_counts"]["val"], 2)
        self.assertEqual(report["sensor_content_integrity"]["file_count"], 20)
        self.assertEqual(len(report["canonical_targets_sha256"]), 64)

    def test_cli_report_is_accepted_by_training_gate(self):
        from autonomous_driving.dbnet_layout import build_labeled_observations
        from autonomous_driving.target_mapping import TargetSchema, map_raw_observations
        from autonomous_driving.preflight_gate import verify_preflight_report

        result = self.run_cli("--hash-sensor-files")
        self.assertEqual(result.returncode, 0, result.stderr)
        schema = TargetSchema(0, 1, "deg", "km/h")
        observations = map_raw_observations(
            build_labeled_observations(self.data), schema
        )
        kwargs = {
            "observations": observations,
            "schema": schema,
            "sequence_length": 4,
            "stride": 1,
            "data_root": self.data,
            "provenance_source": "https://example.org/synthetic-test-only",
            "require_sensor_hashes": True,
        }
        self.assertTrue(verify_preflight_report(self.output, **kwargs))
        (self.data / "train/1/dvr_66x200/0.jpg").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Camera or LiDAR"):
            verify_preflight_report(self.output, **kwargs)

    def test_cli_rejects_report_inside_raw_dataset(self):
        unsafe_output = self.data / "train" / "preflight-report.json"
        result = self.run_cli("--output", str(unsafe_output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the raw dataset", result.stderr)
        self.assertFalse(unsafe_output.exists())

    def test_cli_rejects_missing_lidar(self):
        (self.data / "train/1/points_16384/2.las").unlink()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(report["readiness"]["passed"])
        self.assertFalse(report["readiness"]["no_missing_sensor_or_label_files"])

    def test_cli_rejects_out_of_range_speed(self):
        (self.data / "val/1/behavior.csv").write_text(
            "".join(f"{i},{200 if i == 3 else 20+i}\n" for i in range(5)),
            encoding="utf-8",
        )
        result = self.run_cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(report["readiness"]["no_target_range_violations"])


if __name__ == "__main__":
    unittest.main()
