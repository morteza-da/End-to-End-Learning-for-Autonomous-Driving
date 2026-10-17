"""Reject absent or mismatched evidence of target-label interpretation."""
import json
import tempfile
import unittest
from pathlib import Path

from autonomous_driving.label_provenance import load_verified_label_provenance
from autonomous_driving.target_mapping import TargetSchema


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "evidence.json"
        self.schema = TargetSchema(0, 1, "deg", "km/h")
        self.record = {
            "source_title": "Verified dataset specification",
            "source_url": "https://example.org/specification",
            "evidence_excerpt": "Example evidence for unit-test only",
            "reviewed_by": "Unit test",
            "review_date": "2026-10-17",
            "steering_channel": 0,
            "speed_channel": 1,
            "steering_unit": "deg",
            "speed_unit": "km/h",
        }

    def save(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return self.path

    def test_matching_record(self):
        self.assertEqual(
            load_verified_label_provenance(self.save(self.record), self.schema),
            self.record
        )

    def test_reject_missing_evidence(self):
        self.record["evidence_excerpt"] = ""
        with self.assertRaises(ValueError):
            load_verified_label_provenance(self.save(self.record), self.schema)

    def test_reject_channel_mismatch(self):
        self.record["speed_channel"] = 3
        with self.assertRaises(ValueError):
            load_verified_label_provenance(self.save(self.record), self.schema)

    def test_reject_unit_mismatch(self):
        self.record["speed_unit"] = "mph"
        with self.assertRaises(ValueError):
            load_verified_label_provenance(self.save(self.record), self.schema)


if __name__ == "__main__":
    unittest.main()
