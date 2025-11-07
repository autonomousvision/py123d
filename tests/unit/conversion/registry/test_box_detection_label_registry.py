import unittest

from py123d.conversion.registry.box_detection_label_registry import BOX_DETECTION_LABEL_REGISTRY, BoxDetectionLabel


class TestBoxDetectionLabelRegistry(unittest.TestCase):

    def test_correct_type(self):
        """Test that all registered box detection labels are of correct type."""
        for label_class in BOX_DETECTION_LABEL_REGISTRY.values():
            self.assertTrue(issubclass(label_class, BoxDetectionLabel))

    def test_initialize_all_labels(self):
        """Test that all registered box detection labels can be initialized."""
        for label_enum_class in BOX_DETECTION_LABEL_REGISTRY.values():
            label_enum_class: BoxDetectionLabel
            for integer in range(len(label_enum_class)):
                label_a = label_enum_class.from_int(integer)
                label_b = label_enum_class(integer)
                self.assertIsInstance(label_a, label_enum_class)
                self.assertIsInstance(label_b, label_enum_class)

    def test_serialize_deserialize(self):
        """Test that all registered box detection labels can be serialized and deserialized."""
        for label_enum_class in BOX_DETECTION_LABEL_REGISTRY.values():
            label_enum_class: BoxDetectionLabel
            for integer in range(len(label_enum_class)):
                label = label_enum_class.from_int(integer)
                serialized_lower = label.serialize(lower=True)
                serialized_upper = label.serialize(lower=False)
                deserialized_lower = label_enum_class.deserialize(serialized_lower)
                deserialized_upper = label_enum_class.deserialize(serialized_upper)
                self.assertEqual(label, deserialized_lower)
                self.assertEqual(label, deserialized_upper)

    def test_to_default(self):
        """Test that all registered box detection labels can be converted to DefaultBoxDetectionLabel."""
        from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel

        for label_enum_class in BOX_DETECTION_LABEL_REGISTRY.values():
            label_enum_class: BoxDetectionLabel
            for integer in range(len(label_enum_class)):
                label = label_enum_class.from_int(integer)
                default_label = label.to_default()
                self.assertIsInstance(default_label, DefaultBoxDetectionLabel)
