from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa

from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.maps.abstract_map import AbstractMap
from py123d.datatypes.maps.gpkg.gpkg_map import get_global_map_api, get_local_map_api
from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.scene.arrow.utils.arrow_getters import (
    get_box_detections_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_vehicle_state_from_arrow_table,
    get_lidar_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from py123d.datatypes.scene.arrow.utils.arrow_metadata_utils import get_log_metadata_from_arrow
from py123d.datatypes.scene.scene_metadata import LogMetadata, SceneExtractionMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.sensors.camera.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar.lidar import LiDAR, LiDARType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3


class ArrowScene(AbstractScene):

    def __init__(
        self,
        arrow_file_path: Union[Path, str],
        scene_extraction_metadata: Optional[SceneExtractionMetadata] = None,
    ) -> None:

        self._arrow_file_path: Path = Path(arrow_file_path)
        self._log_metadata: LogMetadata = get_log_metadata_from_arrow(arrow_file_path)

        with pa.memory_map(str(self._arrow_file_path), "r") as source:
            reader = pa.ipc.open_file(source)     
            table = reader.read_all()             
            num_rows = table.num_rows
            initial_uuid = table['uuid'][0].as_py()
           
        if scene_extraction_metadata is None:
            scene_extraction_metadata = SceneExtractionMetadata(
                initial_uuid=initial_uuid,
                initial_idx=0,
                duration_s=self._log_metadata.timestep_seconds * num_rows,
                history_s=0.0,
                iteration_duration_s=self._log_metadata.timestep_seconds,
            )
        self._scene_extraction_metadata: SceneExtractionMetadata = scene_extraction_metadata

        # NOTE: Lazy load a log-specific map API, and keep reference.
        # Global maps are LRU cached internally.
        self._local_map_api: Optional[AbstractMap] = None

    ####################################################################################################################
    # Helpers for ArrowScene
    ####################################################################################################################

    def __reduce__(self):
        """Helper for pickling the object."""
        return (
            self.__class__,
            (
                self._arrow_file_path,
                self._scene_extraction_metadata,
            ),
        )

    def _get_recording_table(self) -> pa.Table:
        """Helper function to return an LRU cached reference to the arrow table."""
        return get_lru_cached_arrow_table(self._arrow_file_path)

    def _get_table_index(self, iteration: int) -> int:
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        table_index = self._scene_extraction_metadata.initial_idx + iteration
        return table_index

    ####################################################################################################################
    # Implementation of AbstractScene
    ####################################################################################################################

    def get_log_metadata(self) -> LogMetadata:
        return self._log_metadata

    def get_scene_extraction_metadata(self) -> SceneExtractionMetadata:
        return self._scene_extraction_metadata

    def get_map_api(self) -> Optional[AbstractMap]:
        map_api: Optional[AbstractMap] = None
        if self.log_metadata.map_metadata is not None:
            if self.log_metadata.map_metadata.map_is_local:
                if self._local_map_api is None:
                    map_api = get_local_map_api(self.log_metadata.split, self.log_name)
                    self._local_map_api = map_api
                else:
                    map_api = self._local_map_api
            else:
                map_api = get_global_map_api(self.log_metadata.dataset, self.log_metadata.location)
        return map_api

    def get_timepoint_at_iteration(self, iteration: int) -> TimePoint:
        return get_timepoint_from_arrow_table(self._get_recording_table(), self._get_table_index(iteration))

    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        return get_ego_vehicle_state_from_arrow_table(
            self._get_recording_table(),
            self._get_table_index(iteration),
            self.log_metadata.vehicle_parameters,
        )

    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionWrapper]:
        return get_box_detections_from_arrow_table(self._get_recording_table(), self._get_table_index(iteration))

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetectionWrapper]:
        return get_traffic_light_detections_from_arrow_table(
            self._get_recording_table(), self._get_table_index(iteration)
        )

    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        route_lane_group_ids: Optional[List[int]] = None
        table = self._get_recording_table()
        if "route_lane_group_ids" in table.column_names:
            route_lane_group_ids = table["route_lane_group_ids"][self._get_table_index(iteration)].as_py()
        return route_lane_group_ids

    def get_camera_at_iteration(self, iteration: int, camera_type: Union[PinholeCameraType, FisheyeMEICameraType]) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
        camera: Optional[Union[PinholeCamera, FisheyeMEICamera]] = None
        if camera_type in self.available_camera_types:
            camera = get_camera_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                camera_type,
                self.log_metadata,
            )
        return camera

    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> Optional[LiDAR]:
        lidar: Optional[LiDAR] = None
        if lidar_type in self.available_lidar_types or lidar_type == LiDARType.LIDAR_MERGED:
            lidar = get_lidar_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                lidar_type,
                self.log_metadata,
            )
        return lidar
