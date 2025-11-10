import unittest
from unittest.mock import MagicMock, patch

from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class TestLogMetadata(unittest.TestCase):

    def test_init_minimal(self):
        """Test LogMetadata initialization with minimal required fields."""
        log_metadata = LogMetadata(
            dataset="test_dataset", split="train", log_name="log_001", location="test_location", timestep_seconds=0.1
        )
        self.assertEqual(log_metadata.dataset, "test_dataset")
        self.assertEqual(log_metadata.split, "train")
        self.assertEqual(log_metadata.log_name, "log_001")
        self.assertEqual(log_metadata.location, "test_location")
        self.assertEqual(log_metadata.timestep_seconds, 0.1)
        self.assertIsNone(log_metadata.vehicle_parameters)
        self.assertIsNone(log_metadata.box_detection_label_class)
        self.assertEqual(log_metadata.pinhole_camera_metadata, {})
        self.assertEqual(log_metadata.fisheye_mei_camera_metadata, {})
        self.assertEqual(log_metadata.lidar_metadata, {})
        self.assertIsNone(log_metadata.map_metadata)

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        log_metadata = LogMetadata(
            dataset="test_dataset", split="train", log_name="log_001", location="test_location", timestep_seconds=0.1
        )
        result = log_metadata.to_dict()
        self.assertEqual(result["dataset"], "test_dataset")
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["log_name"], "log_001")
        self.assertEqual(result["location"], "test_location")
        self.assertEqual(result["timestep_seconds"], 0.1)
        self.assertIsNone(result["vehicle_parameters"])
        self.assertIsNone(result["box_detection_label_class"])
        self.assertEqual(result["pinhole_camera_metadata"], {})

    def test_from_dict_minimal(self):
        """Test from_dict with minimal fields."""
        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "vehicle_parameters": None,
            "box_detection_label_class": None,
            "map_metadata": None,
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        self.assertEqual(log_metadata.dataset, "test_dataset")
        self.assertEqual(log_metadata.split, "train")
        self.assertIsNone(log_metadata.vehicle_parameters)

    @patch.object(VehicleParameters, "from_dict")
    def test_from_dict_with_vehicle_parameters(self, mock_vehicle_params):
        """Test from_dict with vehicle parameters."""
        mock_vehicle = MagicMock()
        mock_vehicle_params.return_value = mock_vehicle

        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "vehicle_parameters": {"some": "data"},
            "box_detection_label_class": None,
            "map_metadata": None,
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        mock_vehicle_params.assert_called_once_with({"some": "data"})
        self.assertEqual(log_metadata.vehicle_parameters, mock_vehicle)

    @patch("py123d.datatypes.metadata.log_metadata.BOX_DETECTION_LABEL_REGISTRY", {"TestLabel": MagicMock})
    def test_from_dict_with_box_detection_label(self):
        """Test from_dict with box detection label class."""
        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "vehicle_parameters": None,
            "box_detection_label_class": "TestLabel",
            "map_metadata": None,
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        self.assertIsNotNone(log_metadata.box_detection_label_class)

    def test_from_dict_with_invalid_box_detection_label(self):
        """Test from_dict with invalid box detection label class."""
        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "vehicle_parameters": None,
            "box_detection_label_class": "InvalidLabel",
            "map_metadata": None,
            "version": "1.0.0",
        }
        with self.assertRaises(ValueError) as context:
            LogMetadata.from_dict(data_dict)
        self.assertIn("Unknown box detection label class", str(context.exception))

    @patch.object(MapMetadata, "from_dict")
    def test_from_dict_with_map_metadata(self, mock_map_metadata):
        """Test from_dict with map metadata."""
        mock_map = MagicMock()
        mock_map_metadata.return_value = mock_map

        data_dict = {
            "dataset": "test_dataset",
            "split": "train",
            "log_name": "log_001",
            "location": "test_location",
            "timestep_seconds": 0.1,
            "vehicle_parameters": None,
            "box_detection_label_class": None,
            "map_metadata": {"some": "data"},
            "version": "1.0.0",
        }
        log_metadata = LogMetadata.from_dict(data_dict)
        mock_map_metadata.assert_called_once_with({"some": "data"})
        self.assertEqual(log_metadata.map_metadata, mock_map)

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = LogMetadata(
            dataset="test_dataset",
            split="train",
            log_name="log_001",
            location="test_location",
            timestep_seconds=0.1,
        )
        data_dict = original.to_dict()
        reconstructed = LogMetadata.from_dict(data_dict)

        self.assertEqual(original.dataset, reconstructed.dataset)
        self.assertEqual(original.split, reconstructed.split)
        self.assertEqual(original.log_name, reconstructed.log_name)
        self.assertEqual(original.location, reconstructed.location)
        self.assertEqual(original.timestep_seconds, reconstructed.timestep_seconds)
