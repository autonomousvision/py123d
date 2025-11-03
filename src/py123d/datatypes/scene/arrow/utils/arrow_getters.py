from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from omegaconf import DictConfig

from py123d.conversion.registry.lidar_index_registry import DefaultLiDARIndex
from py123d.conversion.sensor_io.lidar.draco_lidar_io import load_lidar_from_draco_binary
from py123d.conversion.sensor_io.lidar.file_lidar_io import load_lidar_pcs_from_file
from py123d.conversion.sensor_io.lidar.laz_lidar_io import load_lidar_from_laz_binary
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
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDAR, LiDARMetadata, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import BoundingBoxSE3, StateSE3, Vector3D
from py123d.script.utils.dataset_path_utils import get_dataset_paths

DATASET_PATHS: DictConfig = get_dataset_paths()
DATASET_SENSOR_ROOT: Dict[str, Path] = {
    "av2-sensor": DATASET_PATHS.av2_sensor_data_root,
    "nuplan": DATASET_PATHS.nuplan_sensor_root,
    "nuscenes": DATASET_PATHS.nuscenes_data_root,
    "wopd": DATASET_PATHS.wopd_data_root,
    "pandaset": DATASET_PATHS.pandaset_data_root,
    "kitti360": DATASET_PATHS.kitti360_data_root,
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
    camera_type: Union[PinholeCameraType, FisheyeMEICameraType],
    log_metadata: LogMetadata,
) -> Union[PinholeCamera, FisheyeMEICamera]:

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

    camera_metadata = log_metadata.pinhole_camera_metadata[camera_type]
    if hasattr(camera_metadata, "mirror_parameter") and camera_metadata.mirror_parameter is not None:
        return FisheyeMEICamera(
            metadata=camera_metadata,
            image=image,
            extrinsic=extrinsic,
        )
    else:
        return PinholeCamera(
            metadata=camera_metadata,
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
    lidar_column_name = (
        f"{LiDARType.LIDAR_MERGED.serialize()}_data"
        if lidar_column_name not in arrow_table.schema.names
        else lidar_column_name
    )
    if lidar_column_name in arrow_table.schema.names:

        lidar_data = arrow_table[lidar_column_name][index].as_py()
        if isinstance(lidar_data, str):
            lidar_pc_dict = load_lidar_pcs_from_file(relative_path=lidar_data, log_metadata=log_metadata, index=index)
            if lidar_type == LiDARType.LIDAR_MERGED:
                # Merge all available LiDAR point clouds into one
                merged_pc = np.vstack(list(lidar_pc_dict.values()))
                lidar = LiDAR(
                    metadata=LiDARMetadata(
                        lidar_type=LiDARType.LIDAR_MERGED,
                        lidar_index=DefaultLiDARIndex,
                        extrinsic=None,
                    ),
                    point_cloud=merged_pc,
                )
            elif lidar_type in lidar_pc_dict:
                lidar = LiDAR(
                    metadata=log_metadata.lidar_metadata[lidar_type],
                    point_cloud=lidar_pc_dict[lidar_type],
                )
        elif isinstance(lidar_data, bytes):
            lidar_metadata = log_metadata.lidar_metadata[lidar_type]
            if lidar_data.startswith(b"DRACO"):
                # NOTE: DRACO only allows XYZ compression, so we need to override the lidar index here.
                lidar_metadata.lidar_index = DefaultLiDARIndex

                lidar = load_lidar_from_draco_binary(lidar_data, lidar_metadata)
            elif lidar_data.startswith(b"LASF"):

                lidar = load_lidar_from_laz_binary(lidar_data, lidar_metadata)
        elif lidar_data is None:
            lidar = None
        else:
            raise NotImplementedError(
                f"Only string file paths or bytes for LiDAR data are supported, got {type(lidar_data)}"
            )

    return lidar
