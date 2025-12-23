import numpy as np
import pytest

from py123d.conversion.registry.lidar_index_registry import LIDAR_INDEX_REGISTRY
from py123d.datatypes.sensors.lidar import LiDAR, LiDARMetadata, LiDARType
from py123d.geometry import PoseSE3


class TestLiDARType:
    """Test LiDARType enum functionality."""

    def test_lidar_type_enum_values(self):
        """Test that LiDARType enum has correct values."""
        assert LiDARType.LIDAR_UNKNOWN.value == 0
        assert LiDARType.LIDAR_MERGED.value == 1
        assert LiDARType.LIDAR_TOP.value == 2
        assert LiDARType.LIDAR_FRONT.value == 3
        assert LiDARType.LIDAR_SIDE_LEFT.value == 4
        assert LiDARType.LIDAR_SIDE_RIGHT.value == 5
        assert LiDARType.LIDAR_BACK.value == 6
        assert LiDARType.LIDAR_DOWN.value == 7

    def test_lidar_type_enum_names(self):
        """Test that LiDARType enum members have correct names."""
        assert LiDARType.LIDAR_UNKNOWN.name == "LIDAR_UNKNOWN"
        assert LiDARType.LIDAR_MERGED.name == "LIDAR_MERGED"
        assert LiDARType.LIDAR_TOP.name == "LIDAR_TOP"
        assert LiDARType.LIDAR_FRONT.name == "LIDAR_FRONT"
        assert LiDARType.LIDAR_SIDE_LEFT.name == "LIDAR_SIDE_LEFT"
        assert LiDARType.LIDAR_SIDE_RIGHT.name == "LIDAR_SIDE_RIGHT"
        assert LiDARType.LIDAR_BACK.name == "LIDAR_BACK"
        assert LiDARType.LIDAR_DOWN.name == "LIDAR_DOWN"

    def test_lidar_type_from_value(self):
        """Test that LiDARType can be created from integer values."""
        assert LiDARType(0) == LiDARType.LIDAR_UNKNOWN
        assert LiDARType(1) == LiDARType.LIDAR_MERGED
        assert LiDARType(2) == LiDARType.LIDAR_TOP
        assert LiDARType(3) == LiDARType.LIDAR_FRONT
        assert LiDARType(4) == LiDARType.LIDAR_SIDE_LEFT
        assert LiDARType(5) == LiDARType.LIDAR_SIDE_RIGHT
        assert LiDARType(6) == LiDARType.LIDAR_BACK
        assert LiDARType(7) == LiDARType.LIDAR_DOWN

    def test_lidar_type_unique_values(self):
        """Test that all LiDARType enum values are unique."""
        values = [member.value for member in LiDARType]
        assert len(values) == len(set(values))

    def test_lidar_type_count(self):
        """Test that LiDARType has expected number of members."""
        assert len(LiDARType) == 8


class TestLiDARMetadata:
    """Test LiDARMetadata functionality."""

    def setup_method(self):
        """Set up test fixtures."""

        self.lidar_name = "TestLiDAR"

        # Get a lidar index class from registry (assuming at least one exists)
        self.lidar_index_class = next(iter(LIDAR_INDEX_REGISTRY.values()))
        self.lidar_type = LiDARType.LIDAR_TOP
        self.extrinsic = PoseSE3.from_list([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_lidar_metadata_creation_with_extrinsic(self):
        """Test creating LiDARMetadata with extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
            extrinsic=self.extrinsic,
        )
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is not None

    def test_lidar_metadata_creation_without_extrinsic(self):
        """Test creating LiDARMetadata without extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is None

    def test_lidar_metadata_to_dict_with_extrinsic(self):
        """Test serializing LiDARMetadata to dict with extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
            extrinsic=self.extrinsic,
        )
        data_dict = metadata.to_dict()
        assert data_dict["lidar_type"] == self.lidar_type.name
        assert data_dict["lidar_index"] == self.lidar_index_class.__name__
        assert data_dict["extrinsic"] is not None
        assert isinstance(data_dict["extrinsic"], list)

    def test_lidar_metadata_to_dict_without_extrinsic(self):
        """Test serializing LiDARMetadata to dict without extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        data_dict = metadata.to_dict()
        assert data_dict["lidar_type"] == self.lidar_type.name
        assert data_dict["lidar_index"] == self.lidar_index_class.__name__
        assert data_dict["extrinsic"] is None

    def test_lidar_metadata_from_dict_with_extrinsic(self):
        """Test deserializing LiDARMetadata from dict with extrinsic."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index_class.__name__,
            "extrinsic": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        }
        metadata = LiDARMetadata.from_dict(data_dict)
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is not None

    def test_lidar_metadata_from_dict_without_extrinsic(self):
        """Test deserializing LiDARMetadata from dict without extrinsic."""
        data_dict = {
            "lidar_name": self.lidar_name,
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index_class.__name__,
            "extrinsic": None,
        }
        metadata = LiDARMetadata.from_dict(data_dict)
        assert metadata.lidar_type == self.lidar_type
        assert metadata.lidar_index == self.lidar_index_class
        assert metadata.extrinsic is None

    def test_lidar_metadata_roundtrip_with_extrinsic(self):
        """Test roundtrip serialization/deserialization with extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
            extrinsic=self.extrinsic,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LiDARMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_type == metadata.lidar_type
        assert restored_metadata.lidar_index == metadata.lidar_index

    def test_lidar_metadata_roundtrip_without_extrinsic(self):
        """Test roundtrip serialization/deserialization without extrinsic."""
        metadata = LiDARMetadata(
            lidar_name=self.lidar_name,
            lidar_type=self.lidar_type,
            lidar_index=self.lidar_index_class,
        )
        data_dict = metadata.to_dict()
        restored_metadata = LiDARMetadata.from_dict(data_dict)
        assert restored_metadata.lidar_type == metadata.lidar_type
        assert restored_metadata.lidar_index == metadata.lidar_index
        assert restored_metadata.extrinsic is None

    def test_lidar_metadata_from_dict_unknown_index_raises_error(self):
        """Test that unknown lidar index raises ValueError."""
        data_dict = {"lidar_type": self.lidar_type.name, "lidar_index": "UnknownLiDARIndex", "extrinsic": None}
        with pytest.raises(ValueError):
            LiDARMetadata.from_dict(data_dict)


class TestLiDAR:
    """Test LiDAR functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Get a lidar index class from registry

        self.lidars = {}
        self.extrinsic = PoseSE3.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        for lidar_index_name, lidar_index_class in LIDAR_INDEX_REGISTRY.items():
            metadata = LiDARMetadata(
                lidar_name=lidar_index_name,
                lidar_type=LiDARType.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            point_cloud = np.random.rand(100, len(lidar_index_class)).astype(np.float32)
            self.lidars[lidar_index_name] = LiDAR(metadata=metadata, point_cloud=point_cloud)

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
        """Test LiDAR with empty point cloud."""
        for lidar_index_class in LIDAR_INDEX_REGISTRY.values():
            metadata = LiDARMetadata(
                lidar_name="EmptyLiDAR",
                lidar_type=LiDARType.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            empty_point_cloud = np.empty((0, len(lidar_index_class)), dtype=np.float32)
            lidar = LiDAR(metadata=metadata, point_cloud=empty_point_cloud)
            assert lidar.xyz.shape == (0, 3)
            assert lidar.xy.shape == (0, 2)

    def test_lidar_with_single_point(self):
        """Test LiDAR with single point."""
        for lidar_index_class in LIDAR_INDEX_REGISTRY.values():
            metadata = LiDARMetadata(
                lidar_name="SinglePointLiDAR",
                lidar_type=LiDARType.LIDAR_TOP,
                lidar_index=lidar_index_class,
                extrinsic=self.extrinsic,
            )
            single_point_cloud = np.random.rand(1, len(lidar_index_class)).astype(np.float32)
            lidar = LiDAR(metadata=metadata, point_cloud=single_point_cloud)
            assert lidar.xyz.shape == (1, 3)
            assert lidar.xy.shape == (1, 2)

    def test_lidar_point_cloud_dtype(self):
        """Test that point cloud maintains float32 dtype."""
        for lidar in self.lidars.values():
            assert lidar.point_cloud.dtype == np.float32
            assert lidar.xyz.dtype == np.float32
            assert lidar.xy.dtype == np.float32
