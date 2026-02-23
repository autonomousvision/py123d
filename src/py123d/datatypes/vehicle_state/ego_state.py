from __future__ import annotations

from typing import Final, Optional

from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionMetadata, BoxDetectionSE2, BoxDetectionSE3
from py123d.datatypes.time.time_point import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE2, DynamicStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import (
    VehicleParameters,
    center_se2_to_rear_axle_se2,
    center_se3_to_rear_axle_se3,
    rear_axle_se2_to_center_se2,
    rear_axle_se3_to_center_se3,
)
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, PoseSE2, PoseSE3

EGO_TRACK_TOKEN: Final[str] = "ego_vehicle"


class EgoStateSE3:
    """The EgoStateSE3 represents the state of the ego vehicle in SE3 (3D space).
    It includes the rear axle pose, vehicle parameters, optional dynamic state,
    optional timestamp, and optional tire steering angle.
    """

    def __init__(
        self,
        rear_axle_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timestamp: Optional[Timestamp] = None,
        tire_steering_angle: Optional[float] = 0.0,
    ):
        """Initialize an :class:`EgoStateSE3` instance.

        :param rear_axle_se3: The pose of the rear axle in SE3.
        :param vehicle_parameters: The parameters of the vehicle.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param timestamp: The timestamp of the state, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        """
        self._rear_axle_se3 = rear_axle_se3
        self._vehicle_parameters = vehicle_parameters
        self._dynamic_state_se3 = dynamic_state_se3
        self._timestamp: Optional[Timestamp] = timestamp
        self._tire_steering_angle: Optional[float] = tire_steering_angle

    @classmethod
    def from_center(
        cls,
        center_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timestamp: Optional[Timestamp] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the center pose.

        :param center_se3: The center pose in SE3.
        :param vehicle_parameters: The parameters of the vehicle.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param timestamp: The timestamp of the state, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """

        rear_axle_se3 = center_se3_to_rear_axle_se3(
            center_se3=center_se3,
            vehicle_parameters=vehicle_parameters,
        )

        # TODO @DanielDauner: Adapt dynamic state from center to rear-axle
        return EgoStateSE3.from_rear_axle(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=dynamic_state_se3,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timestamp: Optional[Timestamp] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:
        """Create an :class:`EgoStateSE3` from the rear axle pose.

        :param rear_axle_se3: The pose of the rear axle in SE3.
        :param vehicle_parameters: The parameters of the vehicle.
        :param dynamic_state_se3: The dynamic state of the vehicle, defaults to None.
        :param timestamp: The timestamp of the state, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An :class:`EgoStateSE3` instance.
        """

        return EgoStateSE3(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=dynamic_state_se3,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def rear_axle_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` of the rear axle in SE3."""
        return self._rear_axle_se3

    @property
    def rear_axle_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the rear axle in SE2."""
        return self._rear_axle_se3.pose_se2

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """The :class:`~py123d.datatypes.vehicle_state.VehicleParameters` of the vehicle."""
        return self._vehicle_parameters

    @property
    def dynamic_state_se3(self) -> Optional[DynamicStateSE3]:
        """The :class:`~py123d.datatypes.vehicle_state.DynamicStateSE3` of the vehicle."""
        return self._dynamic_state_se3

    @property
    def timestamp(self) -> Optional[Timestamp]:
        """The :class:`~py123d.datatypes.time.TimePoint` of the ego state, if available."""
        return self._timestamp

    @property
    def tire_steering_angle(self) -> Optional[float]:
        """The tire steering angle of the ego state, if available."""
        return self._tire_steering_angle

    @property
    def center_se3(self) -> PoseSE3:
        """The :class:`~py123d.geometry.PoseSE3` of the vehicle center in SE3."""
        return rear_axle_se3_to_center_se3(
            rear_axle_se3=self._rear_axle_se3,
            vehicle_parameters=self._vehicle_parameters,
        )

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the vehicle center in SE2."""
        return self.center_se3.pose_se2

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        """The :class:`~py123d.geometry.BoundingBoxSE3` of the ego vehicle."""
        return BoundingBoxSE3(
            center_se3=self.center_se3,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
            height=self.vehicle_parameters.height,
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return self.bounding_box_se3.bounding_box_se2

    @property
    def box_detection_se3(self) -> BoxDetectionSE3:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE3` projection of the ego vehicle."""
        return BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.EGO,
                timestamp=self.timestamp,
                track_token=EGO_TRACK_TOKEN,
                num_lidar_points=None,
            ),
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=self.dynamic_state_se3.velocity_3d if self.dynamic_state_se3 else None,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of the ego vehicle."""
        return self.box_detection_se3.box_detection_se2

    @property
    def ego_state_se2(self) -> EgoStateSE2:
        """The :class:`EgoStateSE2` projection of this SE3 ego state."""
        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_se2,
            vehicle_parameters=self.vehicle_parameters,
            dynamic_state_se2=self.dynamic_state_se3.dynamic_state_se2 if self.dynamic_state_se3 else None,
            timestamp=self.timestamp,
            tire_steering_angle=self.tire_steering_angle,
        )


class EgoStateSE2:
    """The EgoStateSE2 represents the state of the ego vehicle in SE2 (2D space).
    It includes the rear axle pose, vehicle parameters, optional dynamic state, and optional timestamp.
    """

    def __init__(
        self,
        rear_axle_se2: PoseSE2,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        timestamp: Optional[Timestamp] = None,
        tire_steering_angle: Optional[float] = 0.0,
    ):
        """Initialize an :class:`EgoStateSE2` instance.

        :param rear_axle_se2: The pose of the rear axle in SE2.
        :param vehicle_parameters: The parameters of the vehicle.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2, defaults to None.
        :param timestamp: The timestamp of the state, defaults to None.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0
        """
        self._rear_axle_se2: PoseSE2 = rear_axle_se2
        self._vehicle_parameters: VehicleParameters = vehicle_parameters
        self._dynamic_state_se2: Optional[DynamicStateSE2] = dynamic_state_se2
        self._timestamp: Optional[Timestamp] = timestamp
        self._tire_steering_angle: Optional[float] = tire_steering_angle

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se2: PoseSE2,
        dynamic_state_se2: DynamicStateSE2,
        vehicle_parameters: VehicleParameters,
        timestamp: Timestamp,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the rear axle pose.

        :param rear_axle_se2: The pose of the rear axle in SE2.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """

        return EgoStateSE2(
            rear_axle_se2=rear_axle_se2,
            dynamic_state_se2=dynamic_state_se2,
            vehicle_parameters=vehicle_parameters,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se2: PoseSE2,
        dynamic_state_se2: DynamicStateSE2,
        vehicle_parameters: VehicleParameters,
        timestamp: Timestamp,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:
        """Create an :class:`EgoStateSE2` from the center pose.

        :param center_se2: The pose of the center in SE2.
        :param dynamic_state_se2: The dynamic state of the vehicle in SE2.
        :param vehicle_parameters: The parameters of the vehicle.
        :param timestamp: The timestamp of the state.
        :param tire_steering_angle: The tire steering angle, defaults to 0.0.
        :return: An instance of :class:`EgoStateSE2`.
        """

        rear_axle_se2 = center_se2_to_rear_axle_se2(
            center_se2=center_se2,
            vehicle_parameters=vehicle_parameters,
        )

        # TODO @DanielDauner: Adapt dynamic state from center to rear-axle
        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=rear_axle_se2,
            dynamic_state_se2=dynamic_state_se2,
            vehicle_parameters=vehicle_parameters,
            timestamp=timestamp,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def rear_axle_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the rear axle in SE2."""
        return self._rear_axle_se2

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """The :class:`~py123d.datatypes.vehicle_state.VehicleParameters` of the vehicle."""
        return self._vehicle_parameters

    @property
    def dynamic_state_se2(self) -> Optional[DynamicStateSE2]:
        """The :class:`~py123d.datatypes.vehicle_state.DynamicStateSE2` of the vehicle."""
        return self._dynamic_state_se2

    @property
    def timestamp(self) -> Optional[Timestamp]:
        """The :class:`~py123d.datatypes.time.TimePoint` of the ego state, if available."""
        return self._timestamp

    @property
    def tire_steering_angle(self) -> Optional[float]:
        """The tire steering angle of the ego state, if available."""
        return self._tire_steering_angle

    @property
    def center_se2(self) -> PoseSE2:
        """The :class:`~py123d.geometry.PoseSE2` of the center in SE2."""
        return rear_axle_se2_to_center_se2(rear_axle_se2=self.rear_axle_se2, vehicle_parameters=self.vehicle_parameters)

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        """The :class:`~py123d.geometry.BoundingBoxSE2` of the ego vehicle."""
        return BoundingBoxSE2(
            center_se2=self.center_se2,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        """The :class:`~py123d.datatypes.detections.BoxDetectionSE2` projection of the ego vehicle."""
        return BoxDetectionSE2(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.EGO,
                timestamp=self.timestamp,
                track_token=EGO_TRACK_TOKEN,
                num_lidar_points=None,
            ),
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.dynamic_state_se2.velocity_2d if self.dynamic_state_se2 else None,
        )
