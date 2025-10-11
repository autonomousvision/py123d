# TODO: rename this file and potentially move somewhere more appropriate.

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from d123.datatypes.detections.detection import (
    BoxDetection,
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionWrapper,
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera.pinhole_camera import PinholeCamera, PinholeCameraType
from d123.datatypes.sensors.lidar.lidar import LiDAR, LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3
from d123.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from d123.geometry import BoundingBoxSE3, StateSE3, Vector3D

DATASET_SENSOR_ROOT: Dict[str, Path] = {
    "nuplan": Path(os.environ["NUPLAN_DATA_ROOT"]) / "nuplan-v1.1" / "sensor_blobs",
    "carla": Path(os.environ["CARLA_DATA_ROOT"]) / "sensor_blobs",
    "av2-sensor": Path(os.environ["AV2_SENSOR_DATA_ROOT"]) / "sensor_mini",
}


def get_timepoint_from_arrow_table(arrow_table: pa.Table, index: int) -> TimePoint:
    return TimePoint.from_us(arrow_table["timestamp"][index].as_py())


def get_ego_vehicle_state_from_arrow_table(
    arrow_table: pa.Table, index: int, vehicle_parameters: VehicleParameters
) -> EgoStateSE3:
    timepoint = get_timepoint_from_arrow_table(arrow_table, index)
    return EgoStateSE3.from_array(
        array=pa.array(arrow_table["ego_state"][index]).to_numpy(),
        vehicle_parameters=vehicle_parameters,
        timepoint=timepoint,
    )


def get_box_detections_from_arrow_table(arrow_table: pa.Table, index: int) -> BoxDetectionWrapper:
    timepoint = get_timepoint_from_arrow_table(arrow_table, index)
    box_detections: List[BoxDetection] = []

    for detection_state, detection_velocity, detection_token, detection_type in zip(
        arrow_table["box_detection_state"][index].as_py(),
        arrow_table["box_detection_velocity"][index].as_py(),
        arrow_table["box_detection_token"][index].as_py(),
        arrow_table["box_detection_type"][index].as_py(),
    ):
        box_detection = BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                detection_type=DetectionType(detection_type),
                timepoint=timepoint,
                track_token=detection_token,
                confidence=None,
            ),
            bounding_box_se3=BoundingBoxSE3.from_array(np.array(detection_state)),
            velocity=Vector3D.from_array(np.array(detection_velocity)) if detection_velocity else None,
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


def get_camera_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_type: PinholeCameraType,
    log_metadata: LogMetadata,
) -> PinholeCamera:

    camera_name = camera_type.serialize()
    table_data = arrow_table[f"{camera_name}_data"][index].as_py()
    extrinsic_values = arrow_table[f"{camera_name}_extrinsic"][index].as_py()
    extrinsic = StateSE3.from_list(extrinsic_values) if extrinsic_values is not None else None

    if table_data is None or extrinsic is None:
        return None

    image: Optional[npt.NDArray[np.uint8]] = None

    if isinstance(table_data, str):
        sensor_root = DATASET_SENSOR_ROOT[log_metadata.dataset]
        full_image_path = sensor_root / table_data
        assert full_image_path.exists(), f"Camera file not found: {full_image_path}"
        image = cv2.imread(str(full_image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(table_data, bytes):
        image = cv2.imdecode(np.frombuffer(table_data, np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError("Only string file paths for camera data are supported.")

    return PinholeCamera(
        metadata=log_metadata.camera_metadata[camera_type],
        image=image,
        extrinsic=extrinsic,
    )


def get_lidar_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    lidar_type: LiDARType,
    log_metadata: LogMetadata,
) -> LiDAR:
    assert (
        lidar_type.serialize() in arrow_table.schema.names
    ), f'"{lidar_type.serialize()}" field not found in Arrow table schema.'
    lidar_data = arrow_table[lidar_type.serialize()][index].as_py()
    lidar_metadata = log_metadata.lidar_metadata[lidar_type]

    if isinstance(lidar_data, str):
        sensor_root = DATASET_SENSOR_ROOT[log_metadata.dataset]
        full_lidar_path = sensor_root / lidar_data
        assert full_lidar_path.exists(), f"LiDAR file not found: {full_lidar_path}"

        # NOTE: We move data specific import into if-else block, to avoid data specific import errors
        if log_metadata.dataset == "nuplan":
            from d123.conversion.nuplan.nuplan_load_sensor import load_nuplan_lidar_from_path

            lidar = load_nuplan_lidar_from_path(full_lidar_path, lidar_metadata)
        elif log_metadata.dataset == "carla":
            from d123.conversion.carla.load_sensor import load_carla_lidar_from_path

            lidar = load_carla_lidar_from_path(full_lidar_path, lidar_metadata)
        elif log_metadata.dataset == "wopd":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Loading LiDAR data for dataset {log_metadata.dataset} is not implemented.")

    else:
        # FIXME: This is a temporary fix for WOPD dataset. The lidar data is stored as a flattened array of float32.
        # Ideally the lidar index should handle the dimension. But for now we hardcode it here.
        lidar_data = np.array(lidar_data, dtype=np.float32).reshape(-1, 3)
        lidar_data = np.concatenate([np.zeros_like(lidar_data), lidar_data], axis=-1)
        if log_metadata.dataset == "wopd":
            lidar = LiDAR(metadata=lidar_metadata, point_cloud=lidar_data.T)
        else:
            raise NotImplementedError("Only string file paths for lidar data are supported.")
    return lidar
