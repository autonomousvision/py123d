from enum import IntEnum

from d123.common.utils.enums import classproperty


class Point2DIndex(IntEnum):
    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class Vector2DIndex(IntEnum):
    X = 0
    Y = 1


class StateSE2Index(IntEnum):
    X = 0
    Y = 1
    YAW = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class Point3DIndex(IntEnum):

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class Vector3DIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2


class StateSE3Index(IntEnum):

    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def ROTATION_XYZ(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)


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


class Corners3DIndex(IntEnum):
    FRONT_LEFT_BOTTOM = 0
    FRONT_RIGHT_BOTTOM = 1
    BACK_RIGHT_BOTTOM = 2
    BACK_LEFT_BOTTOM = 3
    FRONT_LEFT_TOP = 4
    FRONT_RIGHT_TOP = 5
    BACK_RIGHT_TOP = 6
    BACK_LEFT_TOP = 7

    @classproperty
    def BOTTOM(cls) -> slice:
        return slice(cls.FRONT_LEFT_BOTTOM, cls.BACK_LEFT_BOTTOM + 1)

    @classproperty
    def TOP(cls) -> slice:
        return slice(cls.FRONT_LEFT_TOP, cls.BACK_LEFT_TOP + 1)
