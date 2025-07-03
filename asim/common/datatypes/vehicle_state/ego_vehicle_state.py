from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt

from asim.common.datatypes.detection.detection import BoxDetection, BoxDetectionSE3, DetectionMetadata
from asim.common.datatypes.detection.detection_types import DetectionType
from asim.common.datatypes.time.time_point import TimePoint
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from asim.common.geometry.vector import Vector3D
from asim.common.utils.enums import classproperty

# TODO: Implement


class EgoVehicleStateIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    LENGTH = 6
    WIDTH = 7
    HEIGHT = 8
    VELOCITY_X = 9
    VELOCITY_Y = 10
    VELOCITY_Z = 11
    ACCELERATION_X = 12
    ACCELERATION_Y = 13
    ACCELERATION_Z = 14
    ANGULAR_VELOCITY_X = 15
    ANGULAR_VELOCITY_Y = 16
    ANGULAR_VELOCITY_Z = 17

    @classproperty
    def BOUNDING_BOX_SE3(cls) -> slice:
        return slice(cls.X, cls.HEIGHT + 1)

    @classproperty
    def DYNAMIC_VEHICLE_STATE(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


@dataclass
class EgoVehicleState:

    bounding_box: BoundingBoxSE3
    dynamic_state: DynamicVehicleState
    timepoint: Optional[TimePoint] = None

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], timepoint: Optional[TimePoint] = None) -> DynamicVehicleState:
        bounding_box = BoundingBoxSE3.from_array(array[EgoVehicleStateIndex.BOUNDING_BOX_SE3])
        dynamic_state = DynamicVehicleState.from_array(array[EgoVehicleStateIndex.DYNAMIC_VEHICLE_STATE])
        return EgoVehicleState(bounding_box, dynamic_state, timepoint)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert the EgoVehicleState to an array.
        :return: An array containing the bounding box and dynamic state information.
        """
        assert isinstance(self.bounding_box, BoundingBoxSE3)
        assert isinstance(self.dynamic_state, DynamicVehicleState)

        bb_array = self.bounding_box.array
        dynamic_array = self.dynamic_state.array

        return np.concatenate((bb_array, dynamic_array), axis=0)

    @property
    def box_detection(self) -> BoxDetection:
        return BoxDetectionSE3(
            metadata=DetectionMetadata(
                detection_type=DetectionType.EGO,
                timepoint=self.timepoint,
                track_token="ego_vehicle",
                confidence=1.0,
            ),
            bounding_box_se3=self.bounding_box,
            velocity=self.dynamic_state.velocity,
        )


class DynamicVehicleStateIndex(IntEnum):

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
    def VELOCITY(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.VELOCITY_Z + 1)

    @classproperty
    def ACCELERATION(cls) -> slice:
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Z + 1)

    @classproperty
    def ANGULAR_VELOCITY(cls) -> slice:
        return slice(cls.ANGULAR_VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


@dataclass
class DynamicVehicleState:
    velocity: Vector3D
    acceleration: Vector3D
    angular_velocity: Vector3D

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> DynamicVehicleState:
        """
        Create a DynamicVehicleState from an array.
        :param array: The array containing the dynamic state information.
        :return: A DynamicVehicleState instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicVehicleStateIndex)
        velocity = Vector3D.from_array(array[DynamicVehicleStateIndex.VELOCITY])
        acceleration = Vector3D.from_array(array[DynamicVehicleStateIndex.ACCELERATION])
        angular_velocity = Vector3D.from_array(array[DynamicVehicleStateIndex.ANGULAR_VELOCITY])
        return DynamicVehicleState(velocity, acceleration, angular_velocity)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert the DynamicVehicleState to an array.
        :return: An array containing the velocity, acceleration, and angular velocity.
        """
        assert isinstance(self.velocity, Vector3D)
        assert isinstance(self.acceleration, Vector3D)
        assert isinstance(self.angular_velocity, Vector3D)

        return np.concatenate(
            (
                self.velocity.array,
                self.acceleration.array,
                self.angular_velocity.array,
            ),
            axis=0,
        )
