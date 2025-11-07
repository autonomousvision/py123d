from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import classproperty
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry import Vector2D, Vector3D


class DynamicStateSE3Index(IntEnum):

    VELOCITY_X = 0
    VELOCITY_Y = 1
    VELOCITY_Z = 2
    ACCELERATION_X = 3
    ACCELERATION_Y = 4
    ACCELERATION_Z = 5
    ANGULAR_VELOCITY_X = 6
    ANGULAR_VELOCITY_Y = 7
    ANGULAR_VELOCITY_Z = 8

    @classproperty
    def VELOCITY_3D(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.VELOCITY_Z + 1)

    @classproperty
    def VELOCITY_2D(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classproperty
    def ACCELERATION_3D(cls) -> slice:
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Z + 1)

    @classproperty
    def ACCELERATION_2D(cls) -> slice:
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)

    @classproperty
    def ANGULAR_VELOCITY_3D(cls) -> slice:
        return slice(cls.ANGULAR_VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


class DynamicStateSE3(ArrayMixin):

    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        velocity: Vector3D,
        acceleration: Vector3D,
        angular_velocity: Vector3D,
    ):
        array = np.zeros(len(DynamicStateSE3Index), dtype=np.float64)
        array[DynamicStateSE3Index.VELOCITY_3D] = velocity.array
        array[DynamicStateSE3Index.ACCELERATION_3D] = acceleration.array
        array[DynamicStateSE3Index.ANGULAR_VELOCITY_3D] = angular_velocity.array
        self._array = array

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> DynamicStateSE3:
        """
        Create a DynamicVehicleState from an array.
        :param array: The array containing the dynamic state information.
        :return: A DynamicVehicleState instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicStateSE3Index)
        instance = object.__new__(cls)
        instance._array = array
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def velocity(self) -> Vector3D:
        return Vector3D.from_array(self._array[DynamicStateSE3Index.VELOCITY_3D], copy=False)

    @property
    def velocity_3d(self) -> Vector3D:
        return self.velocity

    @property
    def velocity_2d(self) -> Vector2D:
        return Vector2D.from_array(self._array[DynamicStateSE3Index.VELOCITY_2D], copy=False)

    @property
    def acceleration(self) -> Vector3D:
        return Vector3D.from_array(self._array[DynamicStateSE3Index.ACCELERATION_3D], copy=False)

    @property
    def acceleration_3d(self) -> Vector3D:
        return self.acceleration

    @property
    def acceleration_2d(self) -> Vector2D:
        return Vector2D.from_array(self._array[DynamicStateSE3Index.ACCELERATION_2D], copy=False)

    @property
    def angular_velocity(self) -> Vector3D:
        return Vector3D.from_array(self._array[DynamicStateSE3Index.ANGULAR_VELOCITY_3D], copy=False)

    @property
    def dynamic_state_se2(self) -> DynamicStateSE2:
        """
        Convert the DynamicVehicleState to a 2D dynamic state.
        :return: A DynamicStateSE2 instance.
        """
        _array = np.zeros(len(DynamicStateSE2Index), dtype=np.float64)
        _array[DynamicStateSE2Index.VELOCITY_2D] = self._array[DynamicStateSE3Index.VELOCITY_2D]
        _array[DynamicStateSE2Index.ACCELERATION_2D] = self._array[DynamicStateSE3Index.ACCELERATION_2D]
        _array[DynamicStateSE2Index.ANGULAR_VELOCITY_Z] = self._array[DynamicStateSE3Index.ANGULAR_VELOCITY_Z]
        return DynamicStateSE2.from_array(_array, copy=False)


class DynamicStateSE2Index(IntEnum):

    VELOCITY_X = 0
    VELOCITY_Y = 1
    ACCELERATION_X = 2
    ACCELERATION_Y = 3
    ANGULAR_VELOCITY_Z = 4

    @classproperty
    def VELOCITY_2D(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classproperty
    def ACCELERATION_2D(cls) -> slice:
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)

    @classproperty
    def ANGULAR_VELOCITY(cls) -> int:
        return cls.ANGULAR_VELOCITY_Z


@dataclass
class DynamicStateSE2(ArrayMixin):

    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        velocity: Vector3D,
        acceleration: Vector3D,
        angular_velocity: Vector3D,
    ):
        array = np.zeros(len(DynamicStateSE3Index), dtype=np.float64)
        array[DynamicStateSE3Index.VELOCITY_3D] = velocity.array
        array[DynamicStateSE3Index.ACCELERATION_3D] = acceleration.array
        array[DynamicStateSE3Index.ANGULAR_VELOCITY_3D] = angular_velocity.array
        self._array = array

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> DynamicStateSE3:
        """
        Create a DynamicVehicleState from an array.
        :param array: The array containing the dynamic state information.
        :return: A DynamicVehicleState instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicStateSE3Index)
        instance = object.__new__(cls)
        instance._array = array
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._array

    @property
    def velocity(self) -> Vector2D:
        return Vector2D.from_array(self._array[DynamicStateSE2Index.VELOCITY_2D], copy=False)

    @property
    def velocity_2d(self) -> Vector2D:
        return self.velocity

    @property
    def acceleration(self) -> Vector2D:
        return Vector2D.from_array(self._array[DynamicStateSE2Index.ACCELERATION_2D], copy=False)

    @property
    def acceleration_2d(self) -> Vector2D:
        return Vector2D.from_array(self._array[DynamicStateSE2Index.ACCELERATION_2D], copy=False)

    @property
    def angular_velocity(self) -> float:
        return self._array[DynamicStateSE2Index.ANGULAR_VELOCITY_Z]
