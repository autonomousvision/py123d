from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from omegaconf import DictConfig

from py123d.datatypes.detections.box_detection_types import BoxDetectionType
from py123d.datatypes.detections.box_detections import (
    BoxDetection,
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionWrapper,
)
from py123d.datatypes.detections.traffic_light_detections import (
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.sensors.lidar.lidar import LiDAR, LiDARType
from py123d.datatypes.sensors.lidar.lidar_index import DefaultLidarIndex
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import BoundingBoxSE3, StateSE3, Vector3D
from py123d.script.utils.dataset_path_utils import get_dataset_paths

DATASET_PATHS: DictConfig = get_dataset_paths()
DATASET_SENSOR_ROOT: Dict[str, Path] = {
    "nuplan": DATASET_PATHS.nuplan_sensor_root,
    "av2-sensor": DATASET_PATHS.av2_sensor_data_root,
    "wopd": DATASET_PATHS.wopd_data_root,
    "pandaset": DATASET_PATHS.pandaset_data_root,
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
                box_detection_type=BoxDetectionType(detection_type),
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
    traffic_light_detections: Optional[List[TrafficLightDetection]] = None

    if "traffic_light_ids" in arrow_table.schema.names and "traffic_light_types" in arrow_table.schema.names:
        traffic_light_detections: List[TrafficLightDetection] = []
        for lane_id, status in zip(
            arrow_table["traffic_light_ids"][index].as_py(),
            arrow_table["traffic_light_types"][index].as_py(),
        ):
            traffic_light_detection = TrafficLightDetection(
                timepoint=timepoint,
                lane_id=lane_id,
                status=TrafficLightStatus(status),
            )
            traffic_light_detections.append(traffic_light_detection)

        traffic_light_detections = TrafficLightDetectionWrapper(traffic_light_detections=traffic_light_detections)

    return traffic_light_detections


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
        assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
        full_image_path = Path(sensor_root) / table_data
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

    lidar: Optional[LiDAR] = None
    lidar_column_name = f"{lidar_type.serialize()}_data"
    if lidar_column_name in arrow_table.schema.names:

        lidar_data = arrow_table[lidar_column_name][index].as_py()
        lidar_metadata = log_metadata.lidar_metadata[lidar_type]

        if isinstance(lidar_data, str):
            sensor_root = DATASET_SENSOR_ROOT[log_metadata.dataset]
            assert (
                sensor_root is not None
            ), f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
            full_lidar_path = Path(sensor_root) / lidar_data
            assert full_lidar_path.exists(), f"LiDAR file not found: {full_lidar_path}"

            # NOTE: We move data specific import into if-else block, to avoid data specific import errors
            if log_metadata.dataset == "nuplan":
                from py123d.conversion.datasets.nuplan.nuplan_load_sensor import load_nuplan_lidar_from_path

                lidar = load_nuplan_lidar_from_path(full_lidar_path, lidar_metadata)

            elif log_metadata.dataset == "carla":
                raise NotImplementedError("Loading LiDAR data for Carla dataset is not implemented.")
            elif log_metadata.dataset == "av2-sensor":
                from py123d.conversion.datasets.av2.utils.av2_sensor_loading import load_av2_sensor_lidar_pc_from_path

                lidar_pc_dict = load_av2_sensor_lidar_pc_from_path(full_lidar_path)

                assert (
                    lidar_type in lidar_pc_dict
                ), f"LiDAR type {lidar_type} not found in AV2 sensor data at {full_lidar_path}."
                lidar = LiDAR(metadata=lidar_metadata, point_cloud=lidar_pc_dict[lidar_type])

            elif log_metadata.dataset == "wopd":

                raise NotImplementedError
            elif log_metadata.dataset == "pandaset":
                from py123d.conversion.datasets.pandaset.pandaset_sensor_loading import (
                    load_pandaset_lidars_pc_from_path,
                )

                ego_state_se3 = get_ego_vehicle_state_from_arrow_table(
                    arrow_table, index, log_metadata.vehicle_parameters
                )

                lidar_pc_dict = load_pandaset_lidars_pc_from_path(full_lidar_path, ego_state_se3)
                assert (
                    lidar_type in lidar_pc_dict
                ), f"LiDAR type {lidar_type} not found in Pandaset data at {full_lidar_path}."
                lidar = LiDAR(metadata=lidar_metadata, point_cloud=lidar_pc_dict[lidar_type])
            else:
                raise NotImplementedError(f"Loading LiDAR data for dataset {log_metadata.dataset} is not implemented.")

        elif isinstance(lidar_data, bytes):

            if lidar_data.startswith(b"DRACO"):
                from py123d.conversion.log_writer.utils.draco_lidar_compression import decompress_lidar_from_draco

                # NOTE: DRACO only allows XYZ compression, so we need to override the lidar index here.
                lidar_metadata.lidar_index = DefaultLidarIndex

                lidar = decompress_lidar_from_draco(lidar_data, lidar_metadata)
            elif lidar_data.startswith(b"LASF"):

                from py123d.conversion.log_writer.utils.laz_lidar_compression import decompress_lidar_from_laz

                lidar = decompress_lidar_from_laz(lidar_data, lidar_metadata)
        elif lidar_data is None:
            lidar = None
        else:
            raise NotImplementedError(
                f"Only string file paths or bytes for LiDAR data are supported, got {type(lidar_data)}"
            )

    return lidar
