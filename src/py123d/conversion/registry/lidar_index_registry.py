from __future__ import annotations

from enum import IntEnum
from typing import Dict, Type

from py123d.common.utils.enums import classproperty

LIDAR_INDEX_REGISTRY: Dict[str, Type[LiDARIndex]] = {}


def register_lidar_index(enum_class):
    """Decorator to register a LiDARIndex enum class."""
    LIDAR_INDEX_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


class LiDARIndex(IntEnum):
    """Base class for all LiDAR Index enums. Defines common indices for LiDAR point clouds."""

    @classproperty
    def XY(self) -> slice:
        """
        Returns a slice for the XY coordinates of the LiDAR point cloud.
        """
        return slice(self.X, self.Y + 1)  # pyright: ignore[reportAttributeAccessIssue]

    @classproperty
    def XYZ(self) -> slice:
        """
        Returns a slice for the XYZ coordinates of the LiDAR point cloud.
        """
        return slice(self.X, self.Z + 1)  # pyright: ignore[reportAttributeAccessIssue]


@register_lidar_index
class DefaultLiDARIndex(LiDARIndex):
    """Default LiDAR indices for XYZ point clouds."""

    X = 0
    Y = 1
    Z = 2


@register_lidar_index
class NuPlanLiDARIndex(LiDARIndex):
    """LiDAR Indexing Scheme for the nuPlan dataset."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4


@register_lidar_index
class CARLALiDARIndex(LiDARIndex):
    """LiDAR Indexing Scheme for the CARLA."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3


@register_lidar_index
class WODPerceptionLiDARIndex(LiDARIndex):
    """Waymo Open Dataset (WOD) - Perception  LiDAR Indexing Scheme, with polar features."""

    RANGE = 0
    INTENSITY = 1
    ELONGATION = 2
    X = 3
    Y = 4
    Z = 5


@register_lidar_index
class KITTI360LiDARIndex(LiDARIndex):
    """KITTI-360 LiDAR Indexing Scheme."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3


@register_lidar_index
class AV2SensorLiDARIndex(LiDARIndex):
    """Argoverse 2 Sensor LiDAR Indexing Scheme."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3


@register_lidar_index
class PandasetLiDARIndex(LiDARIndex):
    """Pandaset LiDAR Indexing Scheme."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3


@register_lidar_index
class NuScenesLiDARIndex(LiDARIndex):
    """NuScenes LiDAR Indexing Scheme."""

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
