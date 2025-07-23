from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt

from d123.common.geometry.base import Point2D, Point3D, Point3DIndex


class Vector2DIndex(IntEnum):
    X = 0
    Y = 1


class Vector2D(Point2D):
    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return float(np.linalg.norm(self.array))

    @property
    def vector_2d(self) -> Vector2D:
        return self


class Vector3DIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2


class Vector3D(Point3D):

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> Vector3D:
        assert array.ndim == 1
        assert array.shape[0] == len(Point3DIndex)
        return cls(array[Point3DIndex.X], array[Point3DIndex.Y], array[Point3DIndex.Z])

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return float(np.linalg.norm(self.array))

    @property
    def vector_2d(self) -> Vector2D:
        return Vector2D(self.x, self.y)
