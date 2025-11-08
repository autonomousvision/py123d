from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionMetadata, BoxDetectionSE2, BoxDetectionSE3
from py123d.datatypes.time.time_point import TimePoint
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

    def __init__(
        self,
        rear_axle_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timepoint: Optional[TimePoint] = None,
        tire_steering_angle: Optional[float] = 0.0,
    ):
        self._rear_axle_se3 = rear_axle_se3
        self._vehicle_parameters = vehicle_parameters
        self._dynamic_state_se3 = dynamic_state_se3
        self._timepoint: Optional[TimePoint] = timepoint
        self._tire_steering_angle: Optional[float] = tire_steering_angle

    @classmethod
    def from_center(
        cls,
        center_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timepoint: Optional[TimePoint] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:

        rear_axle_se3 = center_se3_to_rear_axle_se3(
            center_se3=center_se3,
            vehicle_parameters=vehicle_parameters,
        )

        # TODO @DanielDauner: Adapt dynamic state from center to rear-axle
        return EgoStateSE3.from_rear_axle(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=dynamic_state_se3,
            timepoint=timepoint,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se3: PoseSE3,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se3: Optional[DynamicStateSE3] = None,
        timepoint: Optional[TimePoint] = None,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE3:

        return EgoStateSE3(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=dynamic_state_se3,
            timepoint=timepoint,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def rear_axle_se3(self) -> PoseSE3:
        return self._rear_axle_se3

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        return self._vehicle_parameters

    @property
    def dynamic_state_se3(self) -> Optional[DynamicStateSE3]:
        return self._dynamic_state_se3

    @property
    def timepoint(self) -> Optional[TimePoint]:
        return self._timepoint

    @property
    def tire_steering_angle(self) -> Optional[float]:
        return self._tire_steering_angle

    @property
    def rear_axle_se2(self) -> PoseSE2:
        return self._rear_axle_se3.pose_se2

    @property
    def rear_axle(self) -> PoseSE3:
        return self._rear_axle_se3

    @property
    def center_se3(self) -> PoseSE3:
        return rear_axle_se3_to_center_se3(
            rear_axle_se3=self._rear_axle_se3,
            vehicle_parameters=self._vehicle_parameters,
        )

    @property
    def center_se2(self) -> PoseSE2:
        return self.center_se3.pose_se2

    @property
    def center(self) -> PoseSE3:
        return self.center_se3

    @property
    def bounding_box_se3(self) -> BoundingBoxSE3:
        return BoundingBoxSE3(
            center=self.center_se3,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
            height=self.vehicle_parameters.height,
        )

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return self.bounding_box.bounding_box_se2

    @property
    def bounding_box(self) -> BoundingBoxSE3:
        return self.bounding_box_se3

    @property
    def box_detection_se3(self) -> BoxDetectionSE3:
        return BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.EGO,
                timepoint=self.timepoint,
                track_token=EGO_TRACK_TOKEN,
                num_lidar_points=None,
            ),
            bounding_box_se3=self.bounding_box,
            velocity=self.dynamic_state_se3.velocity,
        )

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        return self.box_detection.box_detection_se2

    @property
    def box_detection(self) -> BoxDetectionSE3:
        return self.box_detection_se3

    @property
    def ego_state_se2(self) -> EgoStateSE2:
        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_se2,
            vehicle_parameters=self.vehicle_parameters,
            dynamic_state_se2=self.dynamic_state_se3.dynamic_state_se2 if self.dynamic_state_se3 else None,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )


@dataclass
class EgoStateSE2:

    def __init__(
        self,
        rear_axle_se2: PoseSE2,
        vehicle_parameters: VehicleParameters,
        dynamic_state_se2: Optional[DynamicStateSE2] = None,
        timepoint: Optional[TimePoint] = None,
        tire_steering_angle: Optional[float] = 0.0,
    ):
        self._rear_axle_se2: PoseSE2 = rear_axle_se2
        self._vehicle_parameters: VehicleParameters = vehicle_parameters
        self._dynamic_state_se2: Optional[DynamicStateSE2] = dynamic_state_se2
        self._timepoint: Optional[TimePoint] = timepoint
        self._tire_steering_angle: Optional[float] = tire_steering_angle

    @classmethod
    def from_rear_axle(
        cls,
        rear_axle_se2: PoseSE2,
        dynamic_state_se2: DynamicStateSE2,
        vehicle_parameters: VehicleParameters,
        timepoint: TimePoint,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:

        return EgoStateSE2(
            rear_axle_se2=rear_axle_se2,
            dynamic_state_se2=dynamic_state_se2,
            vehicle_parameters=vehicle_parameters,
            timepoint=timepoint,
            tire_steering_angle=tire_steering_angle,
        )

    @classmethod
    def from_center(
        cls,
        center_se2: PoseSE2,
        dynamic_state_se2: DynamicStateSE2,
        vehicle_parameters: VehicleParameters,
        timepoint: TimePoint,
        tire_steering_angle: float = 0.0,
    ) -> EgoStateSE2:

        rear_axle_se2 = center_se2_to_rear_axle_se2(
            center_se2=center_se2,
            vehicle_parameters=vehicle_parameters,
        )

        # TODO @DanielDauner: Adapt dynamic state from center to rear-axle
        return EgoStateSE2.from_rear_axle(
            rear_axle_se2=rear_axle_se2,
            dynamic_state_se2=dynamic_state_se2,
            vehicle_parameters=vehicle_parameters,
            timepoint=timepoint,
            tire_steering_angle=tire_steering_angle,
        )

    @property
    def rear_axle_se2(self) -> PoseSE2:
        return self._rear_axle_se2

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        return self._vehicle_parameters

    @property
    def dynamic_state_se2(self) -> Optional[DynamicStateSE3]:
        return self._dynamic_state_se2

    @property
    def timepoint(self) -> Optional[TimePoint]:
        return self._timepoint

    @property
    def tire_steering_angle(self) -> Optional[float]:
        return self._tire_steering_angle

    @property
    def rear_axle(self) -> PoseSE2:
        return self.rear_axle_se2

    @property
    def center_se2(self) -> PoseSE2:
        return rear_axle_se2_to_center_se2(rear_axle_se2=self.rear_axle_se2, vehicle_parameters=self.vehicle_parameters)

    @property
    def center(self) -> PoseSE3:
        return self.center_se2

    @property
    def bounding_box_se2(self) -> BoundingBoxSE2:
        return BoundingBoxSE2(
            center=self.center_se2,
            length=self.vehicle_parameters.length,
            width=self.vehicle_parameters.width,
        )

    @property
    def bounding_box(self) -> BoundingBoxSE2:
        return self.bounding_box_se2

    @property
    def box_detection_se2(self) -> BoxDetectionSE2:
        return BoxDetectionSE2(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.EGO,
                timepoint=self.timepoint,
                track_token=EGO_TRACK_TOKEN,
                num_lidar_points=None,
            ),
            bounding_box_se2=self.bounding_box,
            velocity=self.dynamic_state_se2.velocity,
        )

    @property
    def box_detection(self) -> BoxDetectionSE2:
        return self.box_detection_se2
