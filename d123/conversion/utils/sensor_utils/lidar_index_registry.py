from enum import IntEnum

from d123.common.utils.enums import classproperty

LIDAR_INDEX_REGISTRY = {}


def register_lidar_index(enum_class):
    LIDAR_INDEX_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


class LiDARIndex(IntEnum):

    @classproperty
    def XY(self) -> slice:
        """
        Returns a slice for the XY coordinates of the LiDAR point cloud.
        """
        return slice(self.X, self.Y + 1)

    @classproperty
    def XYZ(self) -> slice:
        """
        Returns a slice for the XYZ coordinates of the LiDAR point cloud.
        """
        return slice(self.X, self.Z + 1)


@register_lidar_index
class DefaultLidarIndex(LiDARIndex):
    X = 0
    Y = 1
    Z = 2


@register_lidar_index
class NuPlanLidarIndex(LiDARIndex):
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
    ID = 5


@register_lidar_index
class CARLALidarIndex(LiDARIndex):
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3


@register_lidar_index
class WOPDLidarIndex(LiDARIndex):
    RANGE = 0
    INTENSITY = 1
    ELONGATION = 2
    X = 3
    Y = 4
    Z = 5
