"""Synthetic end-to-end preflight integration without DBNet or model training."""
import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.dataset_fingerprint import fingerprint_windows
from autonomous_driving.label_quality import audit_target_ranges, require_plausible_targets
from autonomous_driving.preflight_gate import verify_preflight_report
from autonomous_driving.sensor_integrity import fingerprint_sensor_files
from autonomous_driving.sequence_manifest import build_sequence_manifest, split_windows
from autonomous_driving.target_fingerprint import fingerprint_targets
from autonomous_driving.target_mapping import TargetSchema, map_raw_observations


class SyntheticPreflightPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.schema = TargetSchema(0, 1, "deg", "km/h")
        self.raw = []
        for split in ("train", "val"):
            for frame in range(3):
                camera = f"{split}/camera/{frame}.jpg"
                lidar = f"{split}/lidar/{frame}.las"
                for relative, content in ((camera, b"camera"), (lidar, b"cloud")):
                    target = self.root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content + bytes([frame]))
                self.raw.append({
                    "sample_id": f"{split}/drive/{frame}",
                    "drive_id": f"{split}/drive",
                    "frame_index": frame,
                    "split": split,
                    "camera_path": camera,
                    "lidar_path": lidar,
                    "raw_label_columns": (str(frame), str(20 + frame)),
                })
        self.observations = map_raw_observations(self.raw, self.schema)
        self.windows = split_windows(build_sequence_manifest(
            self.observations, sequence_length=2, stride=1
        ))
        self.report = {
            "dataset_root": str(self.root.resolve()),
            "provenance_source": "https://example.org/verified-source",
            "readiness": {"passed": True},
            "target_schema": {
                "steering_channel": 0, "speed_channel": 1,
                "steering_unit": "deg", "speed_unit": "km/h",
            },
            "sequence_length": 2,
            "stride": 1,
            "plausibility_limits": {
                "steering_abs_deg": 40.0,
                "speed_min_kmh": 0,
                "speed_max_kmh": 120.0,
            },
            "target_ranges": audit_target_ranges(self.observations),
            "canonical_targets_sha256": fingerprint_targets(self.observations),
            "sensor_content_integrity": fingerprint_sensor_files(
                self.root, self.observations
            ),
            "window_counts": {
                key: len(value) for key, value in self.windows.items()
            },
            "window_membership_sha256": {
                key: fingerprint_windows(value) if value else None
                for key, value in self.windows.items()
            },
        }
        self.report_path = self.root / "preflight.json"

    def verify(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        return verify_preflight_report(
            self.report_path, observations=self.observations,
            schema=self.schema, sequence_length=2, stride=1,
            data_root=self.root,
            provenance_source="https://example.org/verified-source",
            require_sensor_hashes=True,
        )

    def test_complete_pipeline_accepts_matching_inputs(self):
        require_plausible_targets(self.report["target_ranges"])
        self.assertEqual(self.report["window_counts"]["train"], 2)
        self.assertEqual(self.report["window_counts"]["val"], 2)
        self.assertTrue(self.verify())

    def test_complete_pipeline_rejects_modified_sensor_bytes(self):
        self.assertTrue(self.verify())
        (self.root / "val/lidar/1.las").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Camera or LiDAR"):
            self.verify()

    def test_complete_pipeline_rejects_modified_labels(self):
        self.assertTrue(self.verify())
        self.observations = map_raw_observations(
            [dict(row, raw_label_columns=("8", row["raw_label_columns"][1]))
             if row["frame_index"] == 0 and row["split"] == "train"
             else row for row in self.raw],
            self.schema,
        )
        with self.assertRaisesRegex(ValueError, "Canonical target"):
            self.verify()

    def test_complete_pipeline_rejects_failed_report(self):
        self.report["readiness"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
