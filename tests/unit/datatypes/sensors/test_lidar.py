import numpy as np
import pytest

from py123d.conversion.registry.lidar_index_registry import LIDAR_INDEX_REGISTRY
from py123d.datatypes.sensors.lidar import Lidar, LidarID, LidarMetadata
from py123d.geometry import PoseSE3


class TestLidarID:
    """Test LidarID enum functionality."""

    def test_lidar_type_enum_values(self):
        """Test that LidarID enum has correct values."""
        assert LidarID.LIDAR_UNKNOWN.value == 0
        assert LidarID.LIDAR_MERGED.value == 1
        assert LidarID.LIDAR_TOP.value == 2
        assert LidarID.LIDAR_FRONT.value == 3
        assert LidarID.LIDAR_SIDE_LEFT.value == 4
        assert LidarID.LIDAR_SIDE_RIGHT.value == 5
        assert LidarID.LIDAR_BACK.value == 6
        assert LidarID.LIDAR_DOWN.value == 7

    def test_lidar_type_enum_names(self):
        """Test that LidarID enum members have correct names."""
        assert LidarID.LIDAR_UNKNOWN.name == "LIDAR_UNKNOWN"
        assert LidarID.LIDAR_MERGED.name == "LIDAR_MERGED"
        assert LidarID.LIDAR_TOP.name == "LIDAR_TOP"
        assert LidarID.LIDAR_FRONT.name == "LIDAR_FRONT"
        assert LidarID.LIDAR_SIDE_LEFT.name == "LIDAR_SIDE_LEFT"
        assert LidarID.LIDAR_SIDE_RIGHT.name == "LIDAR_SIDE_RIGHT"
        assert LidarID.LIDAR_BACK.name == "LIDAR_BACK"
        assert LidarID.LIDAR_DOWN.name == "LIDAR_DOWN"

    def test_lidar_type_from_value(self):
        """Test that LidarID can be created from integer values."""
        assert LidarID(0) == LidarID.LIDAR_UNKNOWN
        assert LidarID(1) == LidarID.LIDAR_MERGED
        assert LidarID(2) == LidarID.LIDAR_TOP
        assert LidarID(3) == LidarID.LIDAR_FRONT
        assert LidarID(4) == LidarID.LIDAR_SIDE_LEFT
        assert LidarID(5) == LidarID.LIDAR_SIDE_RIGHT
        assert LidarID(6) == LidarID.LIDAR_BACK
        assert LidarID(7) == LidarID.LIDAR_DOWN

    def test_lidar_type_unique_values(self):
        """Test that all LidarID enum values are unique."""
        values = [member.value for member in LidarID]
        assert len(values) == len(set(values))

    def test_lidar_type_count(self):
        """Test that LidarID has expected number of members."""
        assert len(LidarID) == 8


class TestLidarMetadata:
    """Test LidarMetadata functionality."""

    def setup_method(self):
        """Set up test fixtures."""

        self.lidar_name = "TestLidar"

        # Get a lidar index class from registry (assuming at least one exists)
        self.lidar_index_class = next(iter(LIDAR_INDEX_REGISTRY.values()))
        self.lidar_type = LidarID.LIDAR_TOP
        self.extrinsic = PoseSE3.from_list([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_lidar_metadata_creation_with_extrinsic(self):
        """Test creating LidarMetadata with extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            extrinsic=self.extrinsic,
        )
        assert metadata.lidar_type == self.lidar_type
        assert metadata.extrinsic is not None

    def test_lidar_metadata_creation_without_extrinsic(self):
        """Test creating LidarMetadata without extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is None

    def test_lidar_metadata_to_dict_with_extrinsic(self):
        """Test serializing LidarMetadata to dict with extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            lidar_index=self.lidar_index_class,
            extrinsic=self.extrinsic,
        )
        data_dict = metadata.to_dict()
        assert data_dict["lidar_type"] == self.lidar_type.name
        assert data_dict["lidar_index"] == self.lidar_index_class.__name__
        assert data_dict["extrinsic"] is not None
        assert isinstance(data_dict["extrinsic"], list)

    def test_lidar_metadata_to_dict_without_extrinsic(self):
        """Test serializing LidarMetadata to dict without extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        data_dict = metadata.to_dict()
        assert data_dict["lidar_type"] == self.lidar_type.name
        assert data_dict["lidar_index"] == self.lidar_index_class.__name__
        assert data_dict["extrinsic"] is None

    def test_lidar_metadata_from_dict_with_extrinsic(self):
        """Test deserializing LidarMetadata from dict with extrinsic."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index_class.__name__,
            "extrinsic": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        }
        metadata = LidarMetadata.from_dict(data_dict)
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is not None

    def test_lidar_metadata_from_dict_without_extrinsic(self):
        """Test deserializing LidarMetadata from dict without extrinsic."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index_class.__name__,
            "extrinsic": None,
        }
        metadata = LidarMetadata.from_dict(data_dict)
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is None

    def test_lidar_metadata_roundtrip_with_extrinsic(self):
        """Test roundtrip serialization/deserialization with extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            lidar_index=self.lidar_index_class,
            extrinsic=self.extrinsic,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LidarMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_type == metadata.lidar_type
        assert restored_metadata.lidar_index == metadata.lidar_index

    def test_lidar_metadata_roundtrip_without_extrinsic(self):
        """Test roundtrip serialization/deserialization without extrinsic."""
        metadata = LidarMetadata(
            lidar_name=self.lidar_name,
            lidar_id=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LidarMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_type == metadata.lidar_type
        assert restored_metadata.lidar_index == metadata.lidar_index
        assert restored_metadata.extrinsic is None

    def test_lidar_metadata_from_dict_unknown_index_raises_error(self):
        """Test that unknown lidar index raises ValueError."""
        data_dict = {"lidar_type": self.lidar_type.name, "lidar_index": "UnknownLidarIndex", "extrinsic": None}
        with pytest.raises(ValueError):
            LidarMetadata.from_dict(data_dict)


class TestLidar:
    """Test Lidar functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Get a lidar index class from registry

        self.lidars = {}
        self.extrinsic = PoseSE3.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        for lidar_index_name, lidar_index_class in LIDAR_INDEX_REGISTRY.items():
            metadata = LidarMetadata(
                lidar_name=lidar_index_name,
                lidar_id=LidarID.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            point_cloud = np.random.rand(100, len(lidar_index_class)).astype(np.float32)
            self.lidars[lidar_index_name] = Lidar(metadata=metadata, point_cloud_3d=point_cloud)

    def test_lidar_xyz_property(self):
        """Test xyz property returns correct shape and values."""
        for lidar in self.lidars.values():
            xyz = lidar.xyz
            assert xyz.shape[0] == lidar.point_cloud.shape[0]
            assert xyz.shape[1] == 3

    def test_lidar_xy_property(self):
        """Test xy property returns correct shape and values."""
        for lidar in self.lidars.values():
            xy = lidar.xy
            assert xy.shape[0] == lidar.point_cloud.shape[0]
            assert xy.shape[1] == 2

    def test_lidar_intensity_property_when_available(self):
        """Test intensity property when INTENSITY attribute exists."""
        for lidar in self.lidars.values():
            intensity = lidar.intensity
            if hasattr(lidar.metadata.lidar_index, "INTENSITY"):
                assert intensity is not None
                assert intensity.shape[0] == lidar.point_cloud.shape[0]
            else:
                assert intensity is None

    def test_lidar_intensity_property_when_not_available(self):
        """Test intensity property returns None when not available."""
        for lidar in self.lidars.values():
            if not hasattr(lidar.metadata.lidar_index, "INTENSITY"):
                assert lidar.intensity is None

    def test_lidar_range_property_when_available(self):
        """Test range property when RANGE attribute exists."""
        for lidar in self.lidars.values():
            range_values = lidar.range
            if hasattr(lidar.metadata.lidar_index, "RANGE"):
                assert range_values is not None
                assert range_values.shape[0] == lidar.point_cloud.shape[0]
            else:
                assert range_values is None

    def test_lidar_range_property_when_not_available(self):
        """Test range property returns None when not available."""
        for lidar in self.lidars.values():
            if not hasattr(lidar.metadata.lidar_index, "RANGE"):
                assert lidar.range is None

    def test_lidar_elongation_property_when_available(self):
        """Test elongation property when ELONGATION attribute exists."""
        for lidar in self.lidars.values():
            elongation = lidar.elongation
            if hasattr(lidar.metadata.lidar_index, "ELONGATION"):
                assert elongation is not None
                assert elongation.shape[0] == lidar.point_cloud.shape[0]
            else:
                assert elongation is None

    def test_lidar_elongation_property_when_not_available(self):
        """Test elongation property returns None when not available."""
        for lidar in self.lidars.values():
            if not hasattr(lidar.metadata.lidar_index, "ELONGATION"):
                assert lidar.elongation is None

    def test_lidar_ring_property_when_available(self):
        """Test ring property when RING attribute exists."""
        for lidar in self.lidars.values():
            ring = lidar.ring
            if hasattr(lidar.metadata.lidar_index, "RING"):
                assert ring is not None
                assert ring.shape[0] == lidar.point_cloud.shape[0]
            else:
                assert ring is None

    def test_lidar_ring_property_when_not_available(self):
        """Test ring property returns None when not available."""
        for lidar in self.lidars.values():
            if not hasattr(lidar.metadata.lidar_index, "RING"):
                assert lidar.ring is None

    def test_lidar_with_empty_point_cloud(self):
        """Test Lidar with empty point cloud."""
        for lidar_index_class in LIDAR_INDEX_REGISTRY.values():
            metadata = LidarMetadata(
                lidar_name="EmptyLidar",
                lidar_id=LidarID.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            empty_point_cloud = np.empty((0, len(lidar_index_class)), dtype=np.float32)
            lidar = Lidar(metadata=metadata, point_cloud_3d=empty_point_cloud)
            assert lidar.xyz.shape == (0, 3)
            assert lidar.xy.shape == (0, 2)

    def test_lidar_with_single_point(self):
        """Test Lidar with single point."""
        for lidar_index_class in LIDAR_INDEX_REGISTRY.values():
            metadata = LidarMetadata(
                lidar_name="SinglePointLidar",
                lidar_id=LidarID.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            single_point_cloud = np.random.rand(1, len(lidar_index_class)).astype(np.float32)
            lidar = Lidar(metadata=metadata, point_cloud_3d=single_point_cloud)
            assert lidar.xyz.shape == (1, 3)
            assert lidar.xy.shape == (1, 2)

    def test_lidar_point_cloud_dtype(self):
        """Test that point cloud maintains float32 dtype."""
        for lidar in self.lidars.values():
            assert lidar.point_cloud.dtype == np.float32
            assert lidar.xyz.dtype == np.float32
            assert lidar.xy.dtype == np.float32
