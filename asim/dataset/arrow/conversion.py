# TODO: rename this file to something more appropriate


import numpy as np
import pyarrow as pa

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from asim.common.time.time_point import TimePoint
from asim.common.vehicle_state.ego_vehicle_state import EgoVehicleState
from asim.dataset.maps.abstract_map import List
from asim.dataset.observation.detection.detection import (
    BoxDetection,
    BoxDetectionSE3,
    BoxDetectionWrapper,
    DetectionMetadata,
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from asim.dataset.observation.detection.detection_types import DetectionType


def get_timepoint_from_arrow_table(arrow_table: pa.Table, index: int) -> TimePoint:
    return TimePoint.from_us(arrow_table["timestamp"][index].as_py())


def get_ego_vehicle_state_from_arrow_table(arrow_table: pa.Table, index: int) -> EgoVehicleState:
    timepoint = get_timepoint_from_arrow_table(arrow_table, index)
    return EgoVehicleState.from_array(
        array=pa.array(arrow_table["ego_states"][index]).to_numpy(),
        timepoint=timepoint,
    )


def get_box_detections_from_arrow_table(arrow_table: pa.Table, index: int) -> BoxDetectionWrapper:
    timepoint = get_timepoint_from_arrow_table(arrow_table, index)
    box_detections: List[BoxDetection] = []

    for detection_state, detection_token, detection_type in zip(
        arrow_table["detections_state"][index].as_py(),
        arrow_table["detections_token"][index].as_py(),
        arrow_table["detections_type"][index].as_py(),
    ):
        box_detection = BoxDetectionSE3(
            metadata=DetectionMetadata(
                detection_type=DetectionType(detection_type),
                timepoint=timepoint,
                track_token=detection_token,
                confidence=None,
            ),
            bounding_box_se3=BoundingBoxSE3.from_array(np.array(detection_state)),
            velocity=None,
        )
        box_detections.append(box_detection)
    return BoxDetectionWrapper(box_detections=box_detections)


def get_traffic_light_detections_from_arrow_table(arrow_table: pa.Table, index: int) -> TrafficLightDetectionWrapper:
    timepoint = get_timepoint_from_arrow_table(arrow_table, index)
    traffic_light_detections: List[TrafficLightDetection] = []

    for lane_id, status in zip(
        arrow_table["traffic_light_ids"][index].as_py(), arrow_table["traffic_light_types"][index].as_py()
    ):
        traffic_light_detection = TrafficLightDetection(
            timepoint=timepoint,
            lane_id=lane_id,
            status=TrafficLightStatus(status),
        )
        traffic_light_detections.append(traffic_light_detection)

    return TrafficLightDetectionWrapper(traffic_light_detections=traffic_light_detections)
