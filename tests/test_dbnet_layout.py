import tempfile
import unittest
from pathlib import Path
from autonomous_driving.dbnet_layout import audit_drive, audit_dataset, build_labeled_observations


class DBNetLayoutTests(unittest.TestCase):
    def test_train_val_test_and_positional_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                drive = root / split / "1"
                (drive / "dvr_66x200").mkdir(parents=True)
                (drive / "points_16384").mkdir()
                for i in (0, 1):
                    (drive / "dvr_66x200" / f"{i}.jpg").touch()
                    (drive / "points_16384" / f"{i}.las").touch()
                if split != "test":
                    (drive / "behavior.csv").write_text("1.5,20\n2.5,21\n")
            audit = audit_dataset(root)
            self.assertEqual(len(audit), 3)
            self.assertEqual([x.label_rows for x in audit], [2, 2, None])
            rows = build_labeled_observations(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["raw_label_columns"], ("1.5", "20"))

    def test_detect_missing_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive = Path(tmp)
            (drive / "dvr_66x200").mkdir()
            (drive / "points_16384").mkdir()
            (drive / "dvr_66x200" / "0.jpg").touch()
            (drive / "points_16384" / "1.las").touch()
            (drive / "behavior.csv").write_text("1,2\n")
            audit = audit_drive(drive, "train")
            self.assertEqual(audit.paired_count, 0)
            self.assertEqual(audit.missing_camera, (1,))
            self.assertEqual(audit.missing_lidar, (0,))


if __name__ == "__main__":
    unittest.main()
