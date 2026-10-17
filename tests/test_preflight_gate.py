"""Preflight gate tests using synthetic observation manifests."""
import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.dataset_fingerprint import fingerprint_windows
from autonomous_driving.preflight_gate import verify_preflight_report
from autonomous_driving.sequence_manifest import Observation, build_sequence_manifest, split_windows
from autonomous_driving.target_mapping import TargetSchema


def observations():
    return [
        Observation(f"{split}/drive/{i}", f"{split}/drive", i, split,
                    f"{split}/camera/{i}.jpg", f"{split}/lidar/{i}.las",
                    float(i), float(i))
        for split in ("train", "val") for i in range(4)
    ]


class PreflightGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "audit.json"
        self.rows = observations()
        self.schema = TargetSchema(0, 1, "deg", "km/h")
        windows = split_windows(build_sequence_manifest(self.rows, sequence_length=2))
        self.report = {
            "readiness": {"passed": True},
            "dataset_root": str(Path(self.tmp.name).resolve()),
            "provenance_source": "https://example.org/source",
            "target_schema": {
                "steering_channel": 0, "speed_channel": 1,
                "steering_unit": "deg", "speed_unit": "km/h",
            },
            "sequence_length": 2,
            "stride": 1,
            "plausibility_limits": {
                "steering_abs_deg": 40.0, "speed_min_kmh": 0,
                "speed_max_kmh": 120.0,
            },
            "window_counts": {k: len(v) for k, v in windows.items()},
            "window_membership_sha256": {
                k: fingerprint_windows(v) if v else None for k, v in windows.items()
            },
        }

    def check(self):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")
        return verify_preflight_report(
            self.path, observations=self.rows, schema=self.schema,
            sequence_length=2, stride=1,
            data_root=self.tmp.name, provenance_source="https://example.org/source"
        )

    def test_accept_matching(self):
        self.assertTrue(self.check())

    def test_reject_failed_preflight(self):
        self.report["readiness"]["passed"] = False
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_windows(self):
        self.report["window_membership_sha256"]["val"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_label_mapping(self):
        self.report["target_schema"]["speed_channel"] = 2
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_dataset_root(self):
        self.report["dataset_root"] = "/unrelated/dataset"
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_evidence_source(self):
        self.report["provenance_source"] = "https://example.org/other"
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_stride(self):
        self.report["stride"] = 2
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
