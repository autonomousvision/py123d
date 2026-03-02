from unittest.mock import MagicMock, patch

import pytest

from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import (
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import PoseSE3


class TestLogMetadata:
    def test_init_minimal(self):
        """Test LogMetadata initialization with minimal required fields."""
        log_metadata = LogMetadata(
            dataset="test_dataset", split="train", log_name="log_001", location="test_location", timestep_seconds=0.1
        )
        assert log_metadata.dataset == "test_dataset"
        assert log_metadata.split == "train"
        assert log_metadata.log_name == "log_001"
        assert log_metadata.location == "test_location"
        assert log_metadata.timestep_seconds == 0.1
        assert log_metadata.vehicle_parameters is None
        assert log_metadata.box_detection_label_class is None
        assert log_metadata.pinhole_camera_metadata == {}
        assert log_metadata.fisheye_mei_camera_metadata == {}
        assert log_metadata.lidar_metadata == {}
        assert log_metadata.map_metadata is None

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        log_metadata = LogMetadata(
            dataset="test_dataset", split="train", log_name="log_001", location="test_location", timestep_seconds=0.1
        )
        result = log_metadata.to_dict()
        assert result["dataset"] == "test_dataset"
        assert result["split"] == "train"
        assert result["log_name"] == "log_001"
        assert result["location"] == "test_location"
        assert result["timestep_seconds"] == 0.1
        assert result["vehicle_parameters"] is None
        assert result["box_detection_label_class"] is None
        assert result["pinhole_camera_metadata"] == {}

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
        assert log_metadata.dataset == "test_dataset"
        assert log_metadata.split == "train"
        assert log_metadata.vehicle_parameters is None

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
        assert log_metadata.vehicle_parameters == mock_vehicle

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
        assert log_metadata.box_detection_label_class is not None

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
        with pytest.raises(ValueError):
            LogMetadata.from_dict(data_dict)

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
        assert log_metadata.map_metadata == mock_map

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

        assert original.dataset == reconstructed.dataset
        assert original.split == reconstructed.split
        assert original.log_name == reconstructed.log_name
        assert original.location == reconstructed.location
        assert original.timestep_seconds == reconstructed.timestep_seconds

    def test_is_instance_of_abstract_metadata(self):
        """LogMetadata is an instance of AbstractMetadata."""
        log_metadata = LogMetadata(
            dataset="test_dataset", split="train", log_name="log_001", location="test_location", timestep_seconds=0.1
        )
        assert isinstance(log_metadata, AbstractMetadata)

    def test_roundtrip_with_pinhole_camera_metadata(self):
        """Roundtrip preserves pinhole camera metadata."""
        intrinsics = PinholeIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        distortion = PinholeDistortion(k1=0.1, k2=0.01, p1=0.001, p2=0.001, k3=0.001)
        cam_metadata = PinholeCameraMetadata(
            camera_name="front_cam",
            camera_id=PinholeCameraID.PCAM_F0,
            intrinsics=intrinsics,
            distortion=distortion,
            width=640,
            height=480,
            camera_to_imu_se3=PoseSE3.identity(),
        )
        original = LogMetadata(
            dataset="test_dataset",
            split="train",
            log_name="log_001",
            location="test_location",
            timestep_seconds=0.1,
            pinhole_camera_metadata={PinholeCameraID.PCAM_F0: cam_metadata},
        )
        reconstructed = LogMetadata.from_dict(original.to_dict())

        assert len(reconstructed.pinhole_camera_metadata) == 1
        restored_cam = reconstructed.pinhole_camera_metadata[PinholeCameraID.PCAM_F0]
        assert restored_cam.camera_name == "front_cam"
        assert restored_cam.camera_id == PinholeCameraID.PCAM_F0
        assert restored_cam.width == 640
        assert restored_cam.height == 480
        assert restored_cam.intrinsics.fx == pytest.approx(500.0)
        assert restored_cam.distortion.k1 == pytest.approx(0.1)

    def test_roundtrip_with_lidar_metadata(self):
        """Roundtrip preserves lidar metadata."""
        lidar_to_imu = PoseSE3.from_list([0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        lidar_meta = LidarMetadata(
            lidar_name="top_lidar",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=lidar_to_imu,
        )
        original = LogMetadata(
            dataset="test_dataset",
            split="val",
            log_name="log_002",
            location="test_location",
            timestep_seconds=0.05,
            lidar_metadata={LidarID.LIDAR_TOP: lidar_meta},
        )
        reconstructed = LogMetadata.from_dict(original.to_dict())

        assert len(reconstructed.lidar_metadata) == 1
        restored_lidar = reconstructed.lidar_metadata[LidarID.LIDAR_TOP]
        assert restored_lidar.lidar_name == "top_lidar"
        assert restored_lidar.lidar_id == LidarID.LIDAR_TOP
        assert restored_lidar.lidar_to_imu_se3 == lidar_to_imu

    def test_roundtrip_with_map_metadata(self):
        """Roundtrip preserves nested MapMetadata."""
        map_meta = MapMetadata(
            dataset="test_dataset",
            location="test_location",
            map_has_z=True,
            map_is_local=True,
            split="train",
            log_name="log_001",
        )
        original = LogMetadata(
            dataset="test_dataset",
            split="train",
            log_name="log_001",
            location="test_location",
            timestep_seconds=0.1,
            map_metadata=map_meta,
        )
        reconstructed = LogMetadata.from_dict(original.to_dict())

        assert reconstructed.map_metadata is not None
        assert reconstructed.map_metadata.dataset == "test_dataset"
        assert reconstructed.map_metadata.map_has_z is True
        assert reconstructed.map_metadata.map_is_local is True
