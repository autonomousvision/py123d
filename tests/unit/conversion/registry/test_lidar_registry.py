from enum import IntEnum

import numpy as np

from py123d.conversion.registry.lidar_index_registry import LIDAR_INDEX_REGISTRY, LidarIndex


class TestLidarRegistry:
    def test_registered_types(self):
        """Test that all registered Lidar types are of correct type."""
        for lidar_class in LIDAR_INDEX_REGISTRY.values():
            assert issubclass(lidar_class, LidarIndex)

    def test_initialize_all_types(self):
        """Test that all registered Lidar types can be initialized."""
        for lidar_enum_class in LIDAR_INDEX_REGISTRY.values():
            lidar_enum_class: LidarIndex
            for integer in range(len(lidar_enum_class)):
                lidar_pc_index = lidar_enum_class(integer)
                assert isinstance(lidar_pc_index, LidarIndex)
                assert isinstance(lidar_pc_index, IntEnum)
                assert isinstance(lidar_pc_index, int)

    def test_xy_slice(self):
        """Test that all registered Lidar types have correct xy slice."""
        for lidar_enum_class in LIDAR_INDEX_REGISTRY.values():
            lidar_enum_class: LidarIndex
            dummy_lidar_pc = np.zeros((42, len(lidar_enum_class)), dtype=np.float32)
            lidar_pc_xy_slice = dummy_lidar_pc[..., lidar_enum_class.XY]
            assert lidar_pc_xy_slice.shape[-1] == 2

    def test_xyz_slice(self):
        """Test that all registered Lidar types have correct xyz slice."""
        for lidar_enum_class in LIDAR_INDEX_REGISTRY.values():
            lidar_enum_class: LidarIndex
            dummy_lidar_pc = np.zeros((42, len(lidar_enum_class)), dtype=np.float32)
            lidar_pc_xyz_slice = dummy_lidar_pc[..., lidar_enum_class.XYZ]
            assert lidar_pc_xyz_slice.shape[-1] == 3
