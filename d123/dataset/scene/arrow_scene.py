import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pyarrow as pa

from d123.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.sensor.camera import Camera, CameraMetadata, CameraType, camera_metadata_dict_from_json
from d123.common.datatypes.sensor.lidar import LiDAR, LiDARMetadata, LiDARType, lidar_metadata_dict_from_json
from d123.common.datatypes.time.time_point import TimePoint
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE3
from d123.common.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from d123.dataset.arrow.conversion import (
    get_box_detections_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_vehicle_state_from_arrow_table,
    get_lidar_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from d123.dataset.arrow.helper import open_arrow_table
from d123.dataset.logs.log_metadata import LogMetadata
from d123.dataset.maps.abstract_map import AbstractMap
from d123.dataset.maps.gpkg.gpkg_map import get_local_map_api, get_map_api_from_names
from d123.dataset.scene.abstract_scene import AbstractScene, SceneExtractionInfo

# TODO: Remove or improve open/close dynamic of Scene object.


def _get_scene_data(
    arrow_file_path: Union[Path, str],
) -> Tuple[LogMetadata, VehicleParameters, Dict[CameraType, CameraMetadata]]:
    """
    Extracts the metadata and vehicle parameters from the arrow file.
    """
    # TODO: consider a better way to read metadata, instead of loading the entire table.
    table = open_arrow_table(arrow_file_path)
    metadata = LogMetadata(**json.loads(table.schema.metadata[b"log_metadata"].decode()))
    vehicle_parameters = VehicleParameters(**json.loads(table.schema.metadata[b"vehicle_parameters"].decode()))

    if b"camera_metadata" in table.schema.metadata:
        camera_metadata = camera_metadata_dict_from_json(table.schema.metadata[b"camera_metadata"].decode())
    else:
        camera_metadata = {}

    if b"lidar_metadata" in table.schema.metadata:
        lidar_metadata = lidar_metadata_dict_from_json(table.schema.metadata[b"lidar_metadata"].decode())
    else:
        lidar_metadata = {}

    del table
    return metadata, vehicle_parameters, camera_metadata, lidar_metadata


class ArrowScene(AbstractScene):
    def __init__(
        self,
        arrow_file_path: Union[Path, str],
        scene_extraction_info: Optional[SceneExtractionInfo] = None,
    ) -> None:

        self._recording_table: pa.Table = None

        (
            _metadata,
            _vehicle_parameters,
            _camera_metadata,
            _lidar_metadata,
        ) = _get_scene_data(arrow_file_path)
        self._metadata: LogMetadata = _metadata
        self._vehicle_parameters: VehicleParameters = _vehicle_parameters
        self._camera_metadata: Dict[CameraType, CameraMetadata] = _camera_metadata
        self._lidar_metadata: Dict[LiDARType, LiDARMetadata] = _lidar_metadata

        self._map_api: Optional[AbstractMap] = None

        self._arrow_log_path = arrow_file_path
        self._scene_extraction_info: SceneExtractionInfo = scene_extraction_info

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._arrow_log_path,
                self._scene_extraction_info,
            ),
        )

    @property
    def map_api(self) -> AbstractMap:
        self._lazy_initialize()
        return self._map_api

    @property
    def log_name(self) -> str:
        return str(self._arrow_log_path.stem)

    @property
    def token(self) -> str:
        self._lazy_initialize()
        return self._recording_table["token"][self._get_table_index(0)].as_py()

    @property
    def log_metadata(self) -> LogMetadata:
        return self._metadata

    @property
    def available_camera_types(self) -> List[CameraType]:
        return list(self._camera_metadata.keys())

    @property
    def available_lidar_types(self) -> List[LiDARType]:
        return list(self._lidar_metadata.keys())

    def _get_table_index(self, iteration: int) -> int:
        self._lazy_initialize()
        assert (
            -self.get_number_of_history_iterations() <= iteration < self.get_number_of_iterations()
        ), "Iteration out of bounds"
        table_index = self._scene_extraction_info.initial_idx + iteration
        return table_index

    def get_number_of_iterations(self) -> int:
        self._lazy_initialize()
        return self._scene_extraction_info.number_of_iterations

    def get_number_of_history_iterations(self) -> int:
        self._lazy_initialize()
        return self._scene_extraction_info.number_of_history_iterations

    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        self._lazy_initialize()
        return get_timepoint_from_arrow_table(self._recording_table, self._get_table_index(iteration))

    def get_ego_state_at_iteration(self, iteration: int) -> EgoStateSE3:
        self._lazy_initialize()
        return get_ego_vehicle_state_from_arrow_table(
            self._recording_table, self._get_table_index(iteration), self._vehicle_parameters
        )

    def get_box_detections_at_iteration(self, iteration: int) -> BoxDetectionWrapper:
        self._lazy_initialize()
        return get_box_detections_from_arrow_table(self._recording_table, self._get_table_index(iteration))

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> TrafficLightDetectionWrapper:
        self._lazy_initialize()
        return get_traffic_light_detections_from_arrow_table(self._recording_table, self._get_table_index(iteration))

    def get_detection_recording_at_iteration(self, iteration: int) -> DetectionRecording:
        return DetectionRecording(
            box_detections=self.get_box_detections_at_iteration(iteration),
            traffic_light_detections=self.get_traffic_light_detections_at_iteration(iteration),
        )

    def get_route_lane_group_ids(self, iteration: int) -> List[int]:
        self._lazy_initialize()
        table_index = self._get_table_index(iteration)
        return self._recording_table["route_lane_group_ids"][table_index].as_py()

    def get_camera_at_iteration(self, iteration: int, camera_type: CameraType) -> Camera:
        self._lazy_initialize()
        assert camera_type in self._camera_metadata, f"Camera type {camera_type} not found in metadata."
        table_index = self._get_table_index(iteration)
        return get_camera_from_arrow_table(
            self._recording_table,
            table_index,
            self._camera_metadata[camera_type],
            self.log_metadata,
        )

    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> LiDAR:
        self._lazy_initialize()
        assert lidar_type in self._lidar_metadata, f"LiDAR type {lidar_type} not found in metadata."
        table_index = self._get_table_index(iteration)
        return get_lidar_from_arrow_table(
            self._recording_table,
            table_index,
            self._lidar_metadata[lidar_type],
            self.log_metadata,
        )

    def _lazy_initialize(self) -> None:
        self.open()

    def open(self) -> None:
        if self._map_api is None:
            try:
                if self._metadata.dataset in ["wopd", "av2-sensor"]:
                    # FIXME:
                    split = str(self._arrow_log_path.parent.name)
                    self._map_api = get_local_map_api(split, self._metadata.log_name)
                else:
                    self._map_api = get_map_api_from_names(self._metadata.dataset, self._metadata.location)
                self._map_api.initialize()
            except Exception as e:
                print(f"Error initializing map API: {e}")
        if self._recording_table is None:
            self._recording_table = open_arrow_table(self._arrow_log_path)
        if self._scene_extraction_info is None:
            self._scene_extraction_info = SceneExtractionInfo(
                initial_idx=0,
                duration_s=self._metadata.timestep_seconds * len(self._recording_table),
                history_s=0.0,
                iteration_duration_s=self._metadata.timestep_seconds,
            )

    def close(self) -> None:
        del self._recording_table
        self._recording_table = None
