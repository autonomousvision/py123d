from __future__ import annotations

from enum import IntEnum

from asim.common.geometry.enum import classproperty


class Point2DIndex(IntEnum):
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


class StateSE3Index(IntEnum):
    # TODO: implement

    X = 0
    Y = 1
    Z = 2
    ROLL = 4
    PITCH = 5
    YAW = 3

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def ROTATION_XYZ(cls) -> slice:
        return slice(cls.YAW, cls.PITCH + 1)
