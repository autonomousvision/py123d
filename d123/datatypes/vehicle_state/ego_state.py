from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Final, Optional

import numpy as np
import numpy.typing as npt

from d123.common.utils.enums import classproperty
from d123.datatypes.detections.detection import BoxDetectionMetadata, BoxDetectionSE2, BoxDetectionSE3
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.vehicle_parameters import (
    VehicleParameters,
    center_se2_to_rear_axle_se2,
    center_se3_to_rear_axle_se3,
    rear_axle_se2_to_center_se2,
    rear_axle_se3_to_center_se3,
)
from d123.geometry import BoundingBoxSE2, BoundingBoxSE3, StateSE2, StateSE3, Vector2D, Vector3D

EGO_TRACK_TOKEN: Final[str] = "ego_vehicle"


class EgoStateSE3Index(IntEnum):

    X = 0
    Y = 1
    Z = 2
    QW = 3
    QX = 4
    QY = 5
    QZ = 6
    VELOCITY_X = 7
    VELOCITY_Y = 8
    VELOCITY_Z = 9
    ACCELERATION_X = 10
    ACCELERATION_Y = 11
    ACCELERATION_Z = 12
    ANGULAR_VELOCITY_X = 13
    ANGULAR_VELOCITY_Y = 14
    ANGULAR_VELOCITY_Z = 15

    @classproperty
    def STATE_SE3(cls) -> slice:
        return slice(cls.X, cls.QZ + 1)

    @classproperty
    def DYNAMIC_VEHICLE_STATE(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)

    @classproperty
    def SCALAR(cls) -> slice:
        return slice(cls.QW, cls.QW + 1)

    @classproperty
    def VECTOR(cls) -> slice:
        return slice(cls.QX, cls.QZ + 1)


@dataclass
class EgoStateSE3:

    center_se3: StateSE3
    dynamic_state_se3: DynamicStateSE3
    vehicle_parameters: VehicleParameters
    timepoint: Optional[TimePoint] = None
    tire_steering_angle: float = 0.0

    @classmethod
    def from_array(
        cls,
        array: npt.NDArray[np.float64],
        vehicle_parameters: VehicleParameters,
        timepoint: Optional[TimePoint] = None,
    ) -> EgoStateSE3:
        state_se3 = StateSE3.from_array(array[EgoStateSE3Index.STATE_SE3])
        dynamic_state = DynamicStateSE3.from_array(array[EgoStateSE3Index.DYNAMIC_VEHICLE_STATE])
        return EgoStateSE3(state_se3, dynamic_state, vehicle_parameters, timepoint)

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se3: StateSE3,
        dynamic_state_se3: DynamicStateSE3,
        vehicle_parameters: VehicleParameters,
        time_point: TimePoint,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:

        return EgoStateSE3(
            center_se3=rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_se3, vehicle_parameters=vehicle_parameters),
            dynamic_state_se3=dynamic_state_se3,  # TODO: Adapt dynamic state rear-axle to center
            vehicle_parameters=vehicle_parameters,
            timepoint=time_point,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert the EgoVehicleState to an array.
        :return: An array containing the bounding box and dynamic state information.
        """
        assert isinstance(self.center_se3, StateSE3)
        assert isinstance(self.dynamic_state_se3, DynamicStateSE3)

        center_array = self.center_se3.array
        dynamic_array = self.dynamic_state_se3.array

        return np.concatenate((center_array, dynamic_array), axis=0)

    @property
    def center(self) -> StateSE3:
        return self.center_se3

    @property
    def rear_axle_se3(self) -> StateSE3:
        return center_se3_to_rear_axle_se3(center_se3=self.center_se3, vehicle_parameters=self.vehicle_parameters)

    @property
    def rear_axle_se2(self) -> StateSE2:
        return self.rear_axle_se3.state_se2

    @property
    def rear_axle(self) -> StateSE3:
        return self.rear_axle_se3

    @cached_property
    def bounding_box(self) -> BoundingBoxSE3:
        return BoundingBoxSE3(
            center=self.center_se3,
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
    def box_detection(self) -> BoxDetectionSE3:
        return BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                detection_type=DetectionType.EGO,
                timepoint=self.timepoint,
                track_token=EGO_TRACK_TOKEN,
                confidence=1.0,
            ),
            bounding_box_se3=self.bounding_box,
            velocity=self.dynamic_state_se3.velocity,
        )

    @property
    def box_detection_se3(self) -> BoxDetectionSE3:
        return self.box_detection

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        return self.box_detection.box_detection_se2

    @property
    def ego_state_se2(self) -> EgoStateSE2:
        return EgoStateSE2(
            center_se2=self.center_se3.state_se2,
            dynamic_state_se2=self.dynamic_state_se3.dynamic_state_se2,
            vehicle_parameters=self.vehicle_parameters,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )


@dataclass
class EgoStateSE2:

    center_se2: StateSE2
    dynamic_state_se2: DynamicStateSE2
    vehicle_parameters: VehicleParameters
    timepoint: Optional[TimePoint] = None
    tire_steering_angle: float = 0.0

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se2: StateSE2,
        dynamic_state_se2: DynamicStateSE2,
        vehicle_parameters: VehicleParameters,
        time_point: TimePoint,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:

        return EgoStateSE2(
            center_se2=rear_axle_se2_to_center_se2(rear_axle_se2=rear_axle_se2, vehicle_parameters=vehicle_parameters),
            dynamic_state_se2=dynamic_state_se2,  # TODO: Adapt dynamic state rear-axle to center
            vehicle_parameters=vehicle_parameters,
            timepoint=time_point,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def center(self) -> StateSE2:
        return self.center_se2

    @property
    def rear_axle_se2(self) -> StateSE2:
        return center_se2_to_rear_axle_se2(center_se2=self.center_se2, vehicle_parameters=self.vehicle_parameters)

    @property
    def rear_axle(self) -> StateSE2:
        return self.rear_axle_se2

    @cached_property
    def bounding_box(self) -> BoundingBoxSE2:
        return BoundingBoxSE2(
            center=self.center_se2,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return self.bounding_box

    @property
    def box_detection(self) -> BoxDetectionSE2:
        return BoxDetectionSE2(
            metadata=BoxDetectionMetadata(
                detection_type=DetectionType.EGO,
                timepoint=self.timepoint,
                track_token=EGO_TRACK_TOKEN,
                confidence=1.0,
            ),
            bounding_box_se2=self.bounding_box_se2,
            velocity=self.dynamic_state_se2.velocity,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        return self.box_detection


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
    def VELOCITY(cls) -> slice:
        return slice(cls.VELOCITY_X, cls.VELOCITY_Z + 1)

    @classproperty
    def ACCELERATION(cls) -> slice:
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Z + 1)

    @classproperty
    def ANGULAR_VELOCITY(cls) -> slice:
        return slice(cls.ANGULAR_VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


@dataclass
class DynamicStateSE3:
    # TODO: Make class array like

    velocity: Vector3D
    acceleration: Vector3D
    angular_velocity: Vector3D

    tire_steering_rate: float = 0.0
    angular_acceleration: float = 0.0

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64]) -> DynamicStateSE3:
        """
        Create a DynamicVehicleState from an array.
        :param array: The array containing the dynamic state information.
        :return: A DynamicVehicleState instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicStateSE3Index)
        velocity = Vector3D.from_array(array[DynamicStateSE3Index.VELOCITY])
        acceleration = Vector3D.from_array(array[DynamicStateSE3Index.ACCELERATION])
        angular_velocity = Vector3D.from_array(array[DynamicStateSE3Index.ANGULAR_VELOCITY])
        return DynamicStateSE3(velocity, acceleration, angular_velocity)

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

    @property
    def dynamic_state_se2(self) -> DynamicStateSE2:
        """
        Convert the DynamicVehicleState to a 2D dynamic state.
        :return: A DynamicStateSE2 instance.
        """
        return DynamicStateSE2(
            velocity=self.velocity.vector_2d,
            acceleration=self.acceleration.vector_2d,
            angular_velocity=self.angular_velocity.z,
            tire_steering_rate=self.tire_steering_rate,
            angular_acceleration=self.angular_acceleration,
        )


@dataclass
class DynamicStateSE2:

    velocity: Vector2D
    acceleration: Vector2D
    angular_velocity: float

    tire_steering_rate: float = 0.0
    angular_acceleration: float = 0.0

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert the DynamicVehicleState to an array.
        :return: An array containing the velocity, acceleration, and angular velocity.
        """
        return np.concatenate((self.velocity.array, self.acceleration.array, np.array([self.angular_velocity])), axis=0)
