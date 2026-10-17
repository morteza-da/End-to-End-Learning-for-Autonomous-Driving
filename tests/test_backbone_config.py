"""Standard-library-only checks for backbone modality assignments."""
import unittest

from autonomous_driving.backbone_config import (
    BackboneAssignment,
    FineTuningPlan,
    get_backbone_assignment,
)


class BackboneConfigTests(unittest.TestCase):
    def test_configuration_a(self):
        a = get_backbone_assignment("A")
        self.assertEqual(a.camera_backbone, "resnext50")
        self.assertEqual(a.lidar_backbone, "xception")
        self.assertEqual(a.image_size, (299, 299))

    def test_configuration_b(self):
        b = get_backbone_assignment("b")
        self.assertEqual(b.camera_backbone, "xception")
        self.assertEqual(b.lidar_backbone, "resnext50")

    def test_invalid_assignment(self):
        with self.assertRaises(ValueError):
            BackboneAssignment("A", "xception", "xception")
        with self.assertRaises(ValueError):
            get_backbone_assignment("C")

    def test_explicit_freeze_plan(self):
        plan = FineTuningPlan(("stem",), ("layer4",))
        self.assertEqual(plan.trainable_module_names, ("layer4",))
        with self.assertRaises(ValueError):
            FineTuningPlan(("stem",), ("stem",))


if __name__ == "__main__":
    unittest.main()
