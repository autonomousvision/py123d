from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry.geometry_index import Vector2DIndex, Vector3DIndex


class Vector2D(ArrayMixin):
    """
    Class to represents 2D vectors, in x, y direction.

    Example:
        >>> v1 = Vector2D(3.0, 4.0)
        >>> v2 = Vector2D(1.0, 2.0)
        >>> v3 = v1 + v2
        >>> v3
        Vector2D(4.0, 6.0)
        >>> v1.array
        array([3., 4.])
        >>> v1.magnitude
        5.0
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float):
        """Initialize Vector2D with x, y components."""
        array = np.zeros(len(Vector2DIndex), dtype=np.float64)
        array[Vector2DIndex.X] = x
        array[Vector2DIndex.Y] = y
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Vector2D:
        """Constructs a Vector2D from a numpy array.

        :param array: Array of shape (2,) representing the vector components [x, y], indexed by \
            :class:`~py123d.geometry.Vector2DIndex`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A Vector2D instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Vector2DIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        """The x component of the vector.

        :return: The x component of the vector.
        """
        return self._array[Vector2DIndex.X]

    @property
    def y(self) -> float:
        """The y component of the vector.

        :return: The y component of the vector.
        """
        return self._array[Vector2DIndex.Y]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the 2D vector.

        :return: A numpy array of shape (2,) containing the vector components [x, y], indexed by \
            :class:`~py123d.geometry.Vector2DIndex`.
        """
        array = np.zeros(len(Vector2DIndex), dtype=np.float64)
        array[Vector2DIndex.X] = self.x
        array[Vector2DIndex.Y] = self.y
        return array

    @property
    def magnitude(self) -> float:
        """Calculates the magnitude (length) of the 2D vector.

        :return: The magnitude of the vector.
        """
        return float(np.linalg.norm(self.array))

    @property
    def vector_2d(self) -> Vector2D:
        """The 2D vector itself. Handy for polymorphism.

        :return: A Vector2D instance representing the 2D vector.
        """
        return self

    def __add__(self, other: Vector2D) -> Vector2D:
        """Adds two 2D vectors.

        :param other: The other vector to add.
        :return: A new Vector2D instance representing the sum.
        """
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        """Subtracts two 2D vectors.

        :param other: The other vector to subtract.
        :return: A new Vector2D instance representing the difference.
        """
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        """Multiplies the 2D vector by a scalar.

        :param scalar: The scalar value to multiply with.
        :return: A new Vector2D instance representing the scaled vector.
        """
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        """Divides the 2D vector by a scalar.

        :param scalar: The scalar value to divide by.
        :return: A new Vector2D instance representing the divided vector.
        """
        return Vector2D(self.x / scalar, self.y / scalar)

    def __iter__(self) -> Iterable[float]:
        """Iterator over vector components."""
        return iter((self.x, self.y))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


class Vector3D(ArrayMixin):
    """
    Class to represents 3D vectors, in x, y, z direction.

    Example:
        >>> v1 = Vector3D(1.0, 2.0, 3.0)
        >>> v2 = Vector3D(4.0, 5.0, 6.0)
        >>> v3 = v1 + v2
        >>> v3
        Vector3D(5.0, 7.0, 9.0)
        >>> v1.array
        array([1., 2., 3.])
        >>> v1.magnitude
        3.7416573867739413
    """

    _array: npt.NDArray[np.float64]

    def __init__(self, x: float, y: float, z: float):
        """Initialize Vector3D with x, y, z components."""
        array = np.zeros(len(Vector3DIndex), dtype=np.float64)
        array[Vector3DIndex.X] = x
        array[Vector3DIndex.Y] = y
        array[Vector3DIndex.Z] = z
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Vector3D:
        """Constructs a Vector3D from a numpy array.

        :param array: Array of shape (3,), indexed by :class:`~py123d.geometry.geometry_index.Vector3DIndex`.
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A Vector3D instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(Vector3DIndex)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def x(self) -> float:
        """The x component of the vector.

        :return: The x component of the vector.
        """
        return self._array[Vector3DIndex.X]

    @property
    def y(self) -> float:
        """The y component of the vector.

        :return: The y component of the vector.
        """
        return self._array[Vector3DIndex.Y]

    @property
    def z(self) -> float:
        """The z component of the vector.

        :return: The z component of the vector.
        """
        return self._array[Vector3DIndex.Z]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Returns the vector components as a numpy array

        :return: A numpy array representing the vector components [x, y, z], indexed by \
            :class:`~py123d.geometry.geometry_index.Vector3DIndex`.
        """
        return self._array

    @property
    def magnitude(self) -> float:
        """Calculates the magnitude (length) of the 3D vector.

        :return: The magnitude of the vector.
        """
        return float(np.linalg.norm(self.array))

    @property
    def vector_2d(self) -> Vector2D:
        """Returns the 2D vector projection (x, y) of the 3D vector.

        :return: A Vector2D instance representing the 2D projection.
        """
        return Vector2D(self.x, self.y)

    def __add__(self, other: Vector3D) -> Vector3D:
        """Adds two 3D vectors.

        :param other: The other vector to add.
        :return: A new Vector2D instance representing the sum.
        """
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        """Subtracts two 3D vectors.

        :param other: The other vector to subtract.
        :return: A new Vector3D instance representing the difference.
        """
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        """Multiplies the 2D vector by a scalar.

        :param scalar: The scalar value to multiply with.
        :return: A new Vector3D instance representing the scaled vector.
        """
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vector3D:
        """Divides the 2D vector by a scalar.

        :param scalar: The scalar value to divide by.
        :return: A new Vector3D instance representing the divided vector.
        """
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __iter__(self) -> Iterable[float]:
        """Iterator over vector components."""
        return iter((self.x, self.y, self.z))

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.z))
