from enum import IntEnum

from py123d.common.utils.enums import classproperty


class Point2DIndex(IntEnum):
    """Indexing enum for array-like representations of 2D points (x,y)."""

    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)


class Vector2DIndex(IntEnum):
    """Indexing enum for array-like representations of 2D vectors (x,y)."""

    X = 0
    Y = 1

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) vector components."""
        return slice(cls.X, cls.Y + 1)


class PoseSE2Index(IntEnum):
    """Indexing enum for array-like representations of SE2 poses (x,y,yaw)."""

    X = 0
    Y = 1
    YAW = 2

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def SE2(cls) -> slice:
        """Slice for accessing (x,y,yaw) pose components."""
        return slice(cls.X, cls.YAW + 1)


class Point3DIndex(IntEnum):
    """Indexing enum for array-like representations of 3D points (x,y,z)."""

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        """Slice for accessing (x,y,z) coordinates."""
        return slice(cls.X, cls.Z + 1)


class Vector3DIndex(IntEnum):
    """Indexing enum for array-like representations of 3D vectors (x,y,z)."""

    X = 0
    Y = 1
    Z = 2

    @classproperty
    def XYZ(cls) -> slice:
        """Slice for accessing (x,y,z) vector components."""
        return slice(cls.X, cls.Z + 1)


class EulerAnglesIndex(IntEnum):
    """Indexing enum for array-like representations of Euler angles (roll,pitch,yaw)."""

    ROLL = 0
    PITCH = 1
    YAW = 2


class QuaternionIndex(IntEnum):
    """Indexing enum for array-like representations of quaternions (qw,qx,qy,qz), scalar-first."""

    QW = 0
    QX = 1
    QY = 2
    QZ = 3

    @classproperty
    def SCALAR(cls) -> int:
        """Index for the scalar part of the quaternion."""
        return cls.QW

    @classproperty
    def VECTOR(cls) -> slice:
        """Slice for accessing the imaginary vector part of the quaternion."""
        return slice(cls.QX, cls.QZ + 1)


class EulerPoseSE3Index(IntEnum):
    """Indexing enum for array-like representations of SE3 states with Euler angles (x,y,z,roll,pitch,yaw).

    Notes
    -----
    Representing a pose with Euler angles is deprecated but left in for testing purposes.

    """

    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        """Slice for accessing (x,y,z) coordinates."""
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def EULER_ANGLES(cls) -> slice:
        """Slice for accessing (roll,pitch,yaw) Euler angles."""
        return slice(cls.ROLL, cls.YAW + 1)


class PoseSE3Index(IntEnum):
    """Indexing enum for array-like representations of SE3 poses (x,y,z,qw,qx,qy,qz)."""

    X = 0
    Y = 1
    Z = 2
    QW = 3
    QX = 4
    QY = 5
    QZ = 6

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def XYZ(cls) -> slice:
        """Slice for accessing (x,y,z) coordinates."""
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def QUATERNION(cls) -> slice:
        """Slice for accessing (qw,qx,qy,qz) quaternion components."""
        return slice(cls.QW, cls.QZ + 1)

    @classproperty
    def SCALAR(cls) -> slice:
        """Slice for accessing the scalar part of the quaternion."""
        return slice(cls.QW, cls.QW + 1)

    @classproperty
    def VECTOR(cls) -> slice:
        """Slice for accessing the vector part of the quaternion."""
        return slice(cls.QX, cls.QZ + 1)


class BoundingBoxSE2Index(IntEnum):
    """Indexing enum for array-like representations of bounding boxes in SE2
    - center point (x,y).
    - yaw rotation.
    - extent (length,width).
    """

    X = 0
    Y = 1
    YAW = 2
    LENGTH = 3
    WIDTH = 4

    @classproperty
    def XY(cls) -> slice:
        """Slice for accessing (x,y) coordinates."""
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def SE2(cls) -> slice:
        """Slice for accessing (x,y,yaw) SE2 representation."""
        return slice(cls.X, cls.YAW + 1)

    @classproperty
    def EXTENT(cls) -> slice:
        """Slice for accessing (length,width) extent."""
        return slice(cls.LENGTH, cls.WIDTH + 1)


class Corners2DIndex(IntEnum):
    """Indexes the corners of a bounding boxes in SE2 in the order: front-left, front-right, back-right, back-left."""

    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    BACK_RIGHT = 2
    BACK_LEFT = 3


class BoundingBoxSE3Index(IntEnum):
    """
    Indexes array-like representations of rotated 3D bounding boxes
    - center point (x,y,z).
    - quaternion rotation (qw,qx,qy,qz).
    - extent (length,width,height).
    """

    X = 0
    Y = 1
    Z = 2
    QW = 3
    QX = 4
    QY = 5
    QZ = 6
    LENGTH = 7
    WIDTH = 8
    HEIGHT = 9

    @classproperty
    def XYZ(cls) -> slice:
        """Slice for accessing (x,y,z) coordinates."""
        return slice(cls.X, cls.Z + 1)

    @classproperty
    def SE3(cls) -> slice:
        """Slice for accessing the full SE3 pose representation."""
        return slice(cls.X, cls.QZ + 1)

    @classproperty
    def QUATERNION(cls) -> slice:
        """Slice for accessing (qw,qx,qy,qz) quaternion components."""
        return slice(cls.QW, cls.QZ + 1)

    @classproperty
    def EXTENT(cls) -> slice:
        """Slice for accessing (length,width,height) extent."""
        return slice(cls.LENGTH, cls.HEIGHT + 1)

    @classproperty
    def SCALAR(cls) -> slice:
        """Slice for accessing the scalar part of the quaternion."""
        return slice(cls.QW, cls.QW + 1)

    @classproperty
    def VECTOR(cls) -> slice:
        """Slice for accessing the vector part of the quaternion."""
        return slice(cls.QX, cls.QZ + 1)


class Corners3DIndex(IntEnum):
    """
    Indexes the corners of a BoundingBoxSE3 in the order: \
        front-left-bottom, front-right-bottom, back-right-bottom, back-left-bottom,\
            front-left-top, front-right-top, back-right-top, back-left-top.

            4------5
            |\\    |\\
            | \\   | \\
            0--\\--1  \\
            \\  \\  \\ \\
        l    \\  7-------6    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\3------2    g
            t      width.     h
             h.               t.
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
        """Slice for accessing the four bottom corners."""
        return slice(cls.FRONT_LEFT_BOTTOM, cls.BACK_LEFT_BOTTOM + 1)

    @classproperty
    def TOP(cls) -> slice:
        """Slice for accessing the four top corners."""
        return slice(cls.FRONT_LEFT_TOP, cls.BACK_LEFT_TOP + 1)
