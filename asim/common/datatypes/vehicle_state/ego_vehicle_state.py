from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

from asim.common.datatypes.detection.detection import BoxDetection, BoxDetectionSE3, DetectionMetadata
from asim.common.datatypes.detection.detection_types import DetectionType
from asim.common.datatypes.time.time_point import TimePoint
from asim.common.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from asim.common.geometry.base import StateSE3
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2, BoundingBoxSE3
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
    VELOCITY_X = 6
    VELOCITY_Y = 7
    VELOCITY_Z = 8
    ACCELERATION_X = 9
    ACCELERATION_Y = 10
    ACCELERATION_Z = 11
    ANGULAR_VELOCITY_X = 12
    ANGULAR_VELOCITY_Y = 13
    ANGULAR_VELOCITY_Z = 14

    @classproperty
    def SE3(cls) -> slice:
        return slice(cls.X, cls.YAW + 1)

    @classproperty
    def DYNAMIC_VEHICLE_STATE(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


@dataclass
class EgoVehicleState:

    center: StateSE3
    dynamic_state: DynamicVehicleState
    vehicle_parameters: VehicleParameters
    timepoint: TimePoint

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.float64],
        vehicle_parameters: VehicleParameters,
        timepoint: Optional[TimePoint] = None,
    ) -> DynamicVehicleState:
        state_se3 = StateSE3.from_array(array[EgoVehicleStateIndex.SE3])
        dynamic_state = DynamicVehicleState.from_array(array[EgoVehicleStateIndex.DYNAMIC_VEHICLE_STATE])
        return EgoVehicleState(state_se3, dynamic_state, vehicle_parameters, timepoint)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert the EgoVehicleState to an array.
        :return: An array containing the bounding box and dynamic state information.
        """
        assert isinstance(self.center, StateSE3)
        assert isinstance(self.dynamic_state, DynamicVehicleState)

        center_array = self.center.array
        dynamic_array = self.dynamic_state.array

        return np.concatenate((center_array, dynamic_array), axis=0)

    @property
    def rear_axle(self) -> StateSE3:
        return self.vehicle_parameters.rear_axle_to_center_longitudinal

    @cached_property
    def bounding_box(self) -> BoundingBoxSE3:
        return BoundingBoxSE3(
            center=self.center,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
            height=self.vehicle_parameters.height,
        )

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        return self.bounding_box

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return self.bounding_box.bounding_box_se2

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

    # TODO: add
    #  - tire_steering_angle
    #  - tire_steering_rate
    #  - angular_accel

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
