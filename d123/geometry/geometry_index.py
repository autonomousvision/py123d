from enum import IntEnum

from d123.common.utils.enums import classproperty


class Point2DIndex(IntEnum):
    """
    Indexes array-like representations of 2D points (x,y).
    """

    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class Vector2DIndex(IntEnum):
    """
    Indexes array-like representations of 2D vectors (x,y).
    """

    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class StateSE2Index(IntEnum):
    """
    Indexes array-like representations of SE2 states (x,y,yaw).
    """

    X = 0
    Y = 1
    YAW = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)


class Point3DIndex(IntEnum):
    """
    Indexes array-like representations of 3D points (x,y,z).
    """

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)


class Vector3DIndex(IntEnum):
    """
    Indexes array-like representations of 3D vectors (x,y,z).
    """

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)


class EulerAnglesIndex(IntEnum):
    """
    Indexes array-like representations of Euler angles (roll,pitch,yaw).
    """

    ROLL = 0
    PITCH = 1
    YAW = 2


class QuaternionIndex(IntEnum):
    """
    Indexes array-like representations of quaternions (qw,qx,qy,qz).
    """

    QW = 0
    QX = 1
    QY = 2
    QZ = 3

    @classproperty
    def SCALAR(cls) -> int:
        return cls.QW

    @classproperty
    def VECTOR(cls) -> slice:
        return slice(cls.QX, cls.QZ + 1)


class StateSE3Index(IntEnum):
    """
    Indexes array-like representations of SE3 states (x,y,z,roll,pitch,yaw).
    TODO: Use quaternions for rotation.
    """

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
    def EULER_ANGLES(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)


class QuaternionSE3Index(IntEnum):
    """
    Indexes array-like representations of SE3 states with quaternions (x,y,z,qw,qx,qy,qz).
    """

    X = 0
    Y = 1
    Z = 2
    QW = 3
    QX = 4
    QY = 5
    QZ = 6

    @classproperty
    def XY(cls) -> slice:
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def QUATERNION(cls) -> slice:
        return slice(cls.QW, cls.QZ + 1)

    @classproperty
    def SCALAR(cls) -> slice:
        return cls.QW

    @classproperty
    def VECTOR(cls) -> slice:
        return slice(cls.QX, cls.QZ + 1)


class BoundingBoxSE2Index(IntEnum):
    """
    Indexes array-like representations of rotated 2D bounding boxes (x,y,yaw,length,width).
    """

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
    """
    Indexes the corners of a BoundingBoxSE2 in the order: front-left, front-right, back-right, back-left.
    """

    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    BACK_RIGHT = 2
    BACK_LEFT = 3


class BoundingBoxSE3Index(IntEnum):
    """
    Indexes array-like representations of rotated 3D bounding boxes (x,y,z,roll,pitch,yaw,length,width,height).
    TODO: Use quaternions for rotation.
    """

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
    def EULER_ANGLES(cls) -> slice:
        return slice(cls.ROLL, cls.YAW + 1)

    @classproperty
    def EXTENT(cls) -> slice:
        return slice(cls.LENGTH, cls.HEIGHT + 1)


class Corners3DIndex(IntEnum):
    """
    Indexes the corners of a BoundingBoxSE3 in the order:
        front-left-bottom, front-right-bottom, back-right-bottom, back-left-bottom,
        front-left-top, front-right-top, back-right-top, back-left-top.
    """

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
