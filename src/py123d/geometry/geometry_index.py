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


class MatrixSO2Index:
    """2D indexing for 2x2 SO2 rotation matrices.

    Provides named column indices for accessing the basis vectors (columns) of a 2x2 rotation matrix::

        [cos -sin]      x_axis = [cos, sin]
        [sin  cos]      y_axis = [-sin, cos]

    Examples:
        >>> import numpy as np
        >>> R = np.eye(2)
        >>> R[MatrixSO2Index.X_AXIS]  # x-axis basis vector
        array([1., 0.])
    """

    @classproperty
    def X_AXIS(cls) -> tuple[slice, int]:
        """Index for the x-axis column, i.e. ``R[:, 0]``."""
        return (slice(None), 0)

    @classproperty
    def Y_AXIS(cls) -> tuple[slice, int]:
        """Index for the y-axis column, i.e. ``R[:, 1]``."""
        return (slice(None), 1)


class MatrixSO3Index:
    """2D indexing for 3x3 SO3 rotation matrices.

    Provides named column indices for accessing the basis vectors (columns) of a 3x3 rotation matrix::

        [r00 r01 r02]      x_axis = R[:, 0]
        [r10 r11 r12]      y_axis = R[:, 1]
        [r20 r21 r22]      z_axis = R[:, 2]

    Examples:
        >>> import numpy as np
        >>> R = np.eye(3)
        >>> R[MatrixSO3Index.Z_AXIS]  # z-axis basis vector
        array([0., 0., 1.])
    """

    @classproperty
    def X_AXIS(cls) -> tuple[slice, int]:
        """Index for the x-axis column, i.e. ``R[:, 0]``."""
        return (slice(None), 0)

    @classproperty
    def Y_AXIS(cls) -> tuple[slice, int]:
        """Index for the y-axis column, i.e. ``R[:, 1]``."""
        return (slice(None), 1)

    @classproperty
    def Z_AXIS(cls) -> tuple[slice, int]:
        """Index for the z-axis column, i.e. ``R[:, 2]``."""
        return (slice(None), 2)


class MatrixSE2Index:
    """2D indexing for 3x3 SE2 transformation matrices.

    Provides named indices for accessing the rotation and translation blocks of a 3x3
    homogeneous transformation matrix::

        [r00 r01  tx]
        [r10 r11  ty]
        [  0   0   1]

    Examples:
        >>> import numpy as np
        >>> matrix = np.eye(3)
        >>> matrix[MatrixSE2Index.ROTATION]  # 2x2 rotation block
        array([[1., 0.],
               [0., 1.]])
        >>> matrix[MatrixSE2Index.TRANSLATION]  # [tx, ty]
        array([0., 0.])
    """

    @classproperty
    def ROTATION(cls) -> tuple[slice, slice]:
        """Index for the 2x2 rotation block, i.e. ``matrix[:2, :2]``."""
        return (slice(0, 2), slice(0, 2))

    @classproperty
    def TRANSLATION(cls) -> tuple[slice, int]:
        """Index for the 2x1 translation column, i.e. ``matrix[:2, 2]``."""
        return (slice(0, 2), 2)


class MatrixSE3Index:
    """2D indexing for 4x4 SE3 transformation matrices.

    Provides named indices for accessing the rotation and translation blocks of a 4x4
    homogeneous transformation matrix::

        [r00 r01 r02  tx]
        [r10 r11 r12  ty]
        [r20 r21 r22  tz]
        [  0   0   0   1]

    Examples:
        >>> import numpy as np
        >>> matrix = np.eye(4)
        >>> matrix[MatrixSE3Index.ROTATION]  # 3x3 rotation block
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> matrix[MatrixSE3Index.TRANSLATION]  # [tx, ty, tz]
        array([0., 0., 0.])
    """

    @classproperty
    def ROTATION(cls) -> tuple[slice, slice]:
        """Index for the 3x3 rotation block, i.e. ``matrix[:3, :3]``."""
        return (slice(0, 3), slice(0, 3))

    @classproperty
    def TRANSLATION(cls) -> tuple[slice, int]:
        """Index for the 3x1 translation column, i.e. ``matrix[:3, 3]``."""
        return (slice(0, 3), 3)
