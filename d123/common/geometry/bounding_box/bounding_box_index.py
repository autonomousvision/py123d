from enum import IntEnum

from d123.common.utils.enums import classproperty


class BoundingBoxSE2Index(IntEnum):
    X = 0
    Y = 1
    YAW = 2
    LENGTH = 3
    WIDTH = 4

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def SE2(cls) -> slice:
        return slice(cls.X, cls.YAW + 1)


class Corners2DIndex(IntEnum):
    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    BACK_RIGHT = 2
    BACK_LEFT = 3


class BoundingBoxSE3Index(IntEnum):
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    LENGTH = 6
    WIDTH = 7
    HEIGHT = 8

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def STATE_SE3(cls) -> slice:
        return slice(cls.X, cls.YAW + 1)

    @classproperty
    def ROTATION_XYZ(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)
