# from __future__ import annotations

# from enum import IntEnum
# from typing import Dict, Type

# from py123d.common.utils.enums import classproperty

# LIDAR_INDEX_REGISTRY: Dict[str, Type[LidarIndex]] = {}


# def register_lidar_index(enum_class):
#     """Decorator to register a LidarIndex enum class."""
#     LIDAR_INDEX_REGISTRY[enum_class.__name__] = enum_class
#     return enum_class


# class LidarIndex(IntEnum):
#     """Base class for all Lidar Index enums. Defines common indices for Lidar point clouds."""

#     @classproperty
#     def XY(self) -> slice:
#         """
#         Returns a slice for the XY coordinates of the Lidar point cloud.
#         """
#         return slice(self.X, self.Y + 1)  # pyright: ignore[reportAttributeAccessIssue]

#     @classproperty
#     def XYZ(self) -> slice:
#         """
#         Returns a slice for the XYZ coordinates of the Lidar point cloud.
#         """
#         return slice(self.X, self.Z + 1)  # pyright: ignore[reportAttributeAccessIssue]


# @register_lidar_index
# class DefaultLidarIndex(LidarIndex):
#     """Default Lidar indices for XYZ point clouds."""

#     X = 0
#     Y = 1
#     Z = 2


# @register_lidar_index
# class NuPlanLidarIndex(LidarIndex):
#     """Lidar Indexing Scheme for the nuPlan dataset."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3
#     RING = 4


# @register_lidar_index
# class CARLALidarIndex(LidarIndex):
#     """Lidar Indexing Scheme for the CARLA."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3


# @register_lidar_index
# class WODPerceptionLidarIndex(LidarIndex):
#     """Waymo Open Dataset (WOD) - Perception  Lidar Indexing Scheme, with polar features."""

#     RANGE = 0
#     INTENSITY = 1
#     ELONGATION = 2
#     X = 3
#     Y = 4
#     Z = 5


# @register_lidar_index
# class KITTI360LidarIndex(LidarIndex):
#     """KITTI-360 Lidar Indexing Scheme."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3


# @register_lidar_index
# class AV2SensorLidarIndex(LidarIndex):
#     """Argoverse 2 Sensor Lidar Indexing Scheme."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3


# @register_lidar_index
# class PandasetLidarIndex(LidarIndex):
#     """Pandaset Lidar Indexing Scheme."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3


# @register_lidar_index
# class NuScenesLidarIndex(LidarIndex):
#     """NuScenes Lidar Indexing Scheme."""

#     X = 0
#     Y = 1
#     Z = 2
#     INTENSITY = 3
#     RING = 4
