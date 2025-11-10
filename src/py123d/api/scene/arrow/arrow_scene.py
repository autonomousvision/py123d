from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa

from py123d.api.map.gpkg.gpkg_map_api import get_global_map_api, get_local_map_api
from py123d.api.map.map_api import MapAPI
from py123d.api.scene.arrow.utils.arrow_getters import (
    get_box_detections_se3_from_arrow_table,
    get_camera_from_arrow_table,
    get_ego_state_se3_from_arrow_table,
    get_lidar_from_arrow_table,
    get_route_lane_group_ids_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from py123d.api.scene.arrow.utils.arrow_metadata_utils import get_log_metadata_from_arrow
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraType
from py123d.datatypes.sensors.lidar import LiDAR, LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3


class ArrowSceneAPI(SceneAPI):

    def __init__(
        self,
        arrow_file_path: Union[Path, str],
        scene_extraction_metadata: Optional[SceneMetadata] = None,
    ) -> None:

        self._arrow_file_path: Path = Path(arrow_file_path)
        self._log_metadata: LogMetadata = get_log_metadata_from_arrow(arrow_file_path)

        with pa.memory_map(str(self._arrow_file_path), "r") as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
            num_rows = table.num_rows
            initial_uuid = table["uuid"][0].as_py()

        if scene_extraction_metadata is None:
            scene_extraction_metadata = SceneMetadata(
                initial_uuid=initial_uuid,
                initial_idx=0,
                duration_s=self._log_metadata.timestep_seconds * num_rows,
                history_s=0.0,
                iteration_duration_s=self._log_metadata.timestep_seconds,
            )
        self._scene_extraction_metadata: SceneMetadata = scene_extraction_metadata

        # NOTE: Lazy load a log-specific map API, and keep reference.
        # Global maps are LRU cached internally.
        self._local_map_api: Optional[MapAPI] = None

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

    def get_scene_metadata(self) -> SceneMetadata:
        return self._scene_extraction_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        map_api: Optional[MapAPI] = None
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
        return get_ego_state_se3_from_arrow_table(
            self._get_recording_table(),
            self._get_table_index(iteration),
            self.log_metadata.vehicle_parameters,
        )

    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionWrapper]:
        return get_box_detections_se3_from_arrow_table(
            self._get_recording_table(),
            self._get_table_index(iteration),
            self.log_metadata,
        )

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetectionWrapper]:
        return get_traffic_light_detections_from_arrow_table(
            self._get_recording_table(), self._get_table_index(iteration)
        )

    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        return get_route_lane_group_ids_from_arrow_table(self._get_recording_table(), self._get_table_index(iteration))

    def get_pinhole_camera_at_iteration(
        self, iteration: int, camera_type: PinholeCameraType
    ) -> Optional[PinholeCamera]:
        pinhole_camera: Optional[PinholeCamera] = None
        if camera_type in self.available_pinhole_camera_types:
            pinhole_camera = get_camera_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                camera_type,
                self.log_metadata,
            )
        return pinhole_camera

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_type: FisheyeMEICameraType
    ) -> Optional[FisheyeMEICamera]:
        fisheye_mei_camera: Optional[FisheyeMEICamera] = None
        if camera_type in self.available_fisheye_mei_camera_types:
            fisheye_mei_camera = get_camera_from_arrow_table(
                self._get_recording_table(),
                self._get_table_index(iteration),
                camera_type,
                self.log_metadata,
            )
        return fisheye_mei_camera

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
