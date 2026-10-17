"""Configuration tests requiring no torch/timm installation."""
import unittest

from autonomous_driving.backbone_extraction import BackboneModelNames


class BackboneNamesTests(unittest.TestCase):
    def test_registry_names(self):
        names = BackboneModelNames()
        self.assertEqual(names.for_architecture("xception"), "xception")
        self.assertEqual(names.for_architecture("resnext50"), "resnext50_32x4d")

    def test_unsupported_architecture(self):
        with self.assertRaises(ValueError):
            BackboneModelNames().for_architecture("mobilenet")


if __name__ == "__main__":
    unittest.main()
