from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from omegaconf import DictConfig

from py123d.common.utils.arrow_column_names import (
    BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN,
    BOX_DETECTIONS_LABEL_COLUMN,
    BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN,
    BOX_DETECTIONS_SE3_COLUMNS,
    BOX_DETECTIONS_TOKEN_COLUMN,
    BOX_DETECTIONS_VELOCITY_3D_COLUMN,
    EGO_DYNAMIC_STATE_SE3_COLUMN,
    EGO_REAR_AXLE_SE3_COLUMN,
    EGO_STATE_SE3_COLUMNS,
    FISHEYE_CAMERA_DATA_COLUMN,
    FISHEYE_CAMERA_EXTRINSIC_COLUMN,
    LIDAR_DATA_COLUMN,
    PINHOLE_CAMERA_DATA_COLUMN,
    PINHOLE_CAMERA_EXTRINSIC_COLUMN,
    ROUTE_LANE_GROUP_IDS_COLUMN,
    SCENARIO_TAGS_COLUMN,
    TIMESTAMP_US_COLUMN,
    TRAFFIC_LIGHTS_COLUMNS,
    TRAFFIC_LIGHTS_LANE_ID_COLUMN,
    TRAFFIC_LIGHTS_STATUS_COLUMN,
)
from py123d.common.utils.mixin import ArrayMixin
from py123d.conversion.registry.lidar_index_registry import DefaultLiDARIndex
from py123d.conversion.sensor_io.camera.jpeg_camera_io import decode_image_from_jpeg_binary, load_image_from_jpeg_file
from py123d.conversion.sensor_io.camera.mp4_camera_io import get_mp4_reader_from_path
from py123d.conversion.sensor_io.lidar.draco_lidar_io import is_draco_binary, load_lidar_from_draco_binary
from py123d.conversion.sensor_io.lidar.file_lidar_io import load_lidar_pcs_from_file
from py123d.conversion.sensor_io.lidar.laz_lidar_io import is_laz_binary, load_lidar_from_laz_binary
from py123d.datatypes.detections.box_detections import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionWrapper,
)
from py123d.datatypes.detections.traffic_light_detections import (
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDAR, LiDARMetadata, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D
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
    assert TIMESTAMP_US_COLUMN in arrow_table.schema.names, "Timestamp column not found in Arrow table."
    return TimePoint.from_us(arrow_table[TIMESTAMP_US_COLUMN][index].as_py())


def get_ego_state_se3_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    vehicle_parameters: VehicleParameters,
) -> Optional[EgoStateSE3]:

    ego_state_se3: Optional[EgoStateSE3] = None
    if _all_columns_in_schema(arrow_table, EGO_STATE_SE3_COLUMNS):
        timepoint = get_timepoint_from_arrow_table(arrow_table, index)
        rear_axle_se3 = PoseSE3.from_list(arrow_table[EGO_REAR_AXLE_SE3_COLUMN][index].as_py())
        ego_state_se3 = EgoStateSE3.from_rear_axle(
            rear_axle_se3=rear_axle_se3,
            vehicle_parameters=vehicle_parameters,
            dynamic_state_se3=_get_optional_array_mixin(
                arrow_table[EGO_DYNAMIC_STATE_SE3_COLUMN][index].as_py(), DynamicStateSE3
            ),
            timepoint=timepoint,
        )
    return ego_state_se3


def get_box_detections_se3_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    log_metadata: LogMetadata,
) -> BoxDetectionWrapper:

    box_detections: Optional[BoxDetectionWrapper] = None
    if _all_columns_in_schema(arrow_table, BOX_DETECTIONS_SE3_COLUMNS):
        timepoint = get_timepoint_from_arrow_table(arrow_table, index)
        box_detections_list: List[BoxDetectionSE3] = []
        box_detection_label_class = log_metadata.box_detection_label_class
        for _bounding_box_se3, _token, _label, _velocity, _num_lidar_points in zip(
            arrow_table[BOX_DETECTIONS_BOUNDING_BOX_SE3_COLUMN][index].as_py(),
            arrow_table[BOX_DETECTIONS_TOKEN_COLUMN][index].as_py(),
            arrow_table[BOX_DETECTIONS_LABEL_COLUMN][index].as_py(),
            arrow_table[BOX_DETECTIONS_VELOCITY_3D_COLUMN][index].as_py(),
            arrow_table[BOX_DETECTIONS_NUM_LIDAR_POINTS_COLUMN][index].as_py(),
        ):
            box_detections_list.append(
                BoxDetectionSE3(
                    metadata=BoxDetectionMetadata(
                        label=box_detection_label_class(_label),
                        track_token=_token,
                        num_lidar_points=_num_lidar_points,
                        timepoint=timepoint,
                    ),
                    bounding_box_se3=BoundingBoxSE3.from_list(_bounding_box_se3),
                    velocity=_get_optional_array_mixin(_velocity, Vector3D),
                )
            )
        box_detections = BoxDetectionWrapper(box_detections=box_detections_list)

    return box_detections


def get_traffic_light_detections_from_arrow_table(arrow_table: pa.Table, index: int) -> TrafficLightDetectionWrapper:
    traffic_lights: Optional[List[TrafficLightDetection]] = None
    if _all_columns_in_schema(arrow_table, TRAFFIC_LIGHTS_COLUMNS):
        timepoint = get_timepoint_from_arrow_table(arrow_table, index)
        traffic_light_detections: List[TrafficLightDetection] = []
        for lane_id, status in zip(
            arrow_table[TRAFFIC_LIGHTS_LANE_ID_COLUMN][index].as_py(),
            arrow_table[TRAFFIC_LIGHTS_STATUS_COLUMN][index].as_py(),
        ):
            traffic_light_detections.append(
                TrafficLightDetection(
                    timepoint=timepoint,
                    lane_id=lane_id,
                    status=TrafficLightStatus(status),
                )
            )
        traffic_lights = TrafficLightDetectionWrapper(traffic_light_detections=traffic_light_detections)
    return traffic_lights


def get_camera_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_type: Union[PinholeCameraType, FisheyeMEICameraType],
    log_metadata: LogMetadata,
) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
    assert isinstance(
        camera_type, (PinholeCameraType, FisheyeMEICameraType)
    ), f"camera_type must be PinholeCameraType or FisheyeMEICameraType, got {type(camera_type)}"

    camera: Optional[Union[PinholeCamera, FisheyeMEICamera]] = None

    camera_name = camera_type.serialize()
    is_pinhole = isinstance(camera_type, PinholeCameraType)

    if is_pinhole:
        camera_data_column = PINHOLE_CAMERA_DATA_COLUMN(camera_name)
        camera_extrinsic_column = PINHOLE_CAMERA_EXTRINSIC_COLUMN(camera_name)
    else:
        camera_data_column = FISHEYE_CAMERA_DATA_COLUMN(camera_name)
        camera_extrinsic_column = FISHEYE_CAMERA_EXTRINSIC_COLUMN(camera_name)

    if _all_columns_in_schema(arrow_table, [camera_data_column, camera_extrinsic_column]):
        table_data = arrow_table[camera_data_column][index].as_py()
        extrinsic_data = arrow_table[camera_extrinsic_column][index].as_py()

        if table_data is not None and extrinsic_data is not None:
            extrinsic = PoseSE3.from_list(extrinsic_data)
            image: Optional[npt.NDArray[np.uint8]] = None

            if isinstance(table_data, str):
                sensor_root = DATASET_SENSOR_ROOT[log_metadata.dataset]
                assert (
                    sensor_root is not None
                ), f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
                full_image_path = Path(sensor_root) / table_data
                assert full_image_path.exists(), f"Camera file not found: {full_image_path}"

                image = load_image_from_jpeg_file(full_image_path)
            elif isinstance(table_data, bytes):
                image = decode_image_from_jpeg_binary(table_data)
            elif isinstance(table_data, int):
                image = _unoptimized_demo_mp4_read(log_metadata, camera_name, table_data)
            else:
                raise NotImplementedError(
                    f"Only string file paths, bytes, or int frame indices are supported for camera data, got {type(table_data)}"
                )
            # extrinsic = PoseSE3.from_list(arrow_table[camera_extrinsic_column][index].as_py())

            if is_pinhole:
                camera_metadata = log_metadata.pinhole_camera_metadata[camera_type]
                camera = PinholeCamera(
                    metadata=camera_metadata,
                    image=image,
                    extrinsic=extrinsic,
                )
            else:
                camera_metadata = log_metadata.fisheye_mei_camera_metadata[camera_type]
                camera = FisheyeMEICamera(
                    metadata=camera_metadata,
                    image=image,
                    extrinsic=extrinsic,
                )

    return camera


def get_lidar_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    lidar_type: LiDARType,
    log_metadata: LogMetadata,
) -> LiDAR:

    lidar: Optional[LiDAR] = None
    # NOTE @DanielDauner: Some LiDAR are stored together and are seperated only during loading.
    # In this case, we need to use the merged LiDAR column name.

    lidar_column_name = LIDAR_DATA_COLUMN(lidar_type.serialize())
    lidar_column_name = (
        LIDAR_DATA_COLUMN(LiDARType.LIDAR_MERGED.serialize())
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
            if is_draco_binary(lidar_data):
                # NOTE: DRACO only allows XYZ compression, so we need to override the lidar index here.
                lidar_metadata.lidar_index = DefaultLiDARIndex

                lidar = load_lidar_from_draco_binary(lidar_data, lidar_metadata)
            elif is_laz_binary(lidar_data):

                lidar = load_lidar_from_laz_binary(lidar_data, lidar_metadata)

            else:
                raise ValueError("LiDAR binary data is neither in Draco nor LAZ format.")
        elif lidar_data is not None:
            raise NotImplementedError(
                f"Only string file paths or bytes for LiDAR data are supported, got {type(lidar_data)}"
            )

    return lidar


def get_route_lane_group_ids_from_arrow_table(arrow_table: pa.Table, index: int) -> Optional[List[int]]:
    route_lane_group_ids: Optional[List[int]] = None
    if _all_columns_in_schema(arrow_table, [ROUTE_LANE_GROUP_IDS_COLUMN]):
        route_lane_group_ids = arrow_table[ROUTE_LANE_GROUP_IDS_COLUMN][index].as_py()
    return route_lane_group_ids


def get_scenario_tags_from_arrow_table(arrow_table: pa.Table, index: int) -> Optional[List[int]]:
    scenario_tags: Optional[List[int]] = None
    if _all_columns_in_schema(arrow_table, [SCENARIO_TAGS_COLUMN]):
        scenario_tags = arrow_table[SCENARIO_TAGS_COLUMN][index].as_py()
    return scenario_tags


def _unoptimized_demo_mp4_read(log_metadata: LogMetadata, camera_name: str, frame_index: int) -> Optional[np.ndarray]:
    """A quick and dirty MP4 reader for testing purposes only. Not optimized for performance."""
    image: Optional[npt.NDArray[np.uint8]] = None

    py123d_sensor_root = Path(DATASET_PATHS.py123d_sensors_root)
    mp4_path = py123d_sensor_root / log_metadata.split / log_metadata.log_name / f"{camera_name}.mp4"
    if mp4_path.exists():
        reader = get_mp4_reader_from_path(str(mp4_path))
        image = reader.get_frame(frame_index)

    return image


def _get_optional_array_mixin(data: Optional[Union[List, npt.NDArray]], cls: Type[ArrayMixin]) -> Optional[ArrayMixin]:
    if data is None:
        return None
    if isinstance(data, list):
        return cls.from_list(data)
    elif isinstance(data, np.ndarray):
        return cls.from_array(data, copy=False)
    else:
        raise ValueError(f"Unsupported data type for ArrayMixin conversion: {type(data)}")


def _all_columns_in_schema(arrow_table: pa.Table, columns: List[str]) -> bool:
    return all(column in arrow_table.schema.names for column in columns)
