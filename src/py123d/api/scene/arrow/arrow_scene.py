from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa

from py123d.api.map.arrow_map_api import get_global_map_api, get_local_map_api
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
from py123d.api.scene.arrow.utils.arrow_metadata_utils import (
    get_log_metadata_from_arrow_schema,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.common.utils.arrow_column_names import UUID_COLUMN
from py123d.common.utils.arrow_file_names import (
    BOX_DETECTIONS_FILE,
    EGO_STATE_FILE,
    FISHEYE_CAMERA_FILE,
    INDEX_FILE,
    LIDAR_FILE,
    PINHOLE_CAMERA_FILE,
    ROUTE_FILE,
    TRAFFIC_LIGHTS_FILE,
)
from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.common.utils.uuid_utils import convert_to_str_uuid
from py123d.datatypes.detections import BoxDetectionWrapper, TrafficLightDetectionWrapper
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors import (
    FisheyeMEICamera,
    FisheyeMEICameraType,
    LiDAR,
    LiDARType,
    PinholeCamera,
    PinholeCameraType,
)
from py123d.datatypes.time import TimePoint
from py123d.datatypes.vehicle_state import EgoStateSE3


def _get_complete_log_scene_metadata(log_dir: Union[Path, str], log_metadata: LogMetadata) -> SceneMetadata:
    """Helper function to get the scene metadata for a complete log from a modular log directory."""
    index_path = Path(log_dir) / INDEX_FILE
    table = get_lru_cached_arrow_table(index_path)
    initial_uuid = convert_to_str_uuid(table[UUID_COLUMN][0].as_py())
    num_rows = table.num_rows
    return SceneMetadata(
        initial_uuid=initial_uuid,
        initial_idx=0,
        duration_s=log_metadata.timestep_seconds * num_rows,
        history_s=0.0,
        iteration_duration_s=log_metadata.timestep_seconds,
    )


@lru_cache(maxsize=3_000)
def _get_lru_cached_log_metadata(log_dir: Union[Path, str]) -> LogMetadata:
    """Helper function to get the LRU cached log metadata from a modular log directory."""
    index_path = Path(log_dir) / INDEX_FILE
    table = get_lru_cached_arrow_table(index_path)
    return get_log_metadata_from_arrow_schema(table.schema)


class ArrowSceneAPI(SceneAPI):
    """Scene API for Arrow-based scenes. Provides access to all data modalities in a modular log directory."""

    def __init__(
        self,
        arrow_log_path: Union[Path, str],
        scene_metadata: Optional[SceneMetadata] = None,
    ) -> None:
        """Initializes the :class:`ArrowSceneAPI`.

        :param arrow_log_path: Path to the log directory containing modality Arrow files.
        :param scene_metadata: Scene metadata, defaults to None
        """

        self._arrow_log_path: Path = Path(arrow_log_path)
        self._scene_metadata: Optional[SceneMetadata] = scene_metadata

        # NOTE: Lazy load a log-specific map API, and keep reference.
        # Global maps are LRU cached internally.
        self._local_map_api: Optional[MapAPI] = None

    # Helper methods
    # ------------------------------------------------------------------------------------------------------------------

    def __reduce__(self):
        """Helper for pickling the object."""
        return (
            self.__class__,
            (
                self._arrow_log_path,
                self._scene_metadata,
            ),
        )

    def _get_index_table(self) -> pa.Table:
        """Returns an LRU cached reference to the index.arrow table."""
        return get_lru_cached_arrow_table(self._arrow_log_path / INDEX_FILE)

    def _get_modality_table(self, file_name: str) -> Optional[pa.Table]:
        """Lazy-load and LRU-cache a modality file within the log directory.

        :param file_name: The file name of the modality (e.g., "EgoState.arrow").
        :return: The Arrow table, or None if the file does not exist.
        """
        file_path = self._arrow_log_path / file_name
        if not file_path.exists():
            return None
        return get_lru_cached_arrow_table(file_path)

    def _get_table_index(self, iteration: int) -> int:
        """Helper function to get the table index for a given iteration."""
        assert -self.number_of_history_iterations <= iteration < self.number_of_iterations, "Iteration out of bounds"
        table_index = self.get_scene_metadata().initial_idx + iteration
        return table_index

    # Implementation of abstract methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return _get_lru_cached_log_metadata(self._arrow_log_path)

    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        if self._scene_metadata is None:
            log_metadata = self.get_log_metadata()
            self._scene_metadata = _get_complete_log_scene_metadata(self._arrow_log_path, log_metadata)
        return self._scene_metadata

    def get_map_api(self) -> Optional[MapAPI]:
        """Inherited, see superclass."""
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
        """Inherited, see superclass."""
        return get_timepoint_from_arrow_table(self._get_index_table(), self._get_table_index(iteration))

    def get_ego_state_at_iteration(self, iteration: int) -> Optional[EgoStateSE3]:
        """Inherited, see superclass."""
        table = self._get_modality_table(EGO_STATE_FILE)
        if table is None:
            return None
        timepoint = self.get_timepoint_at_iteration(iteration)
        return get_ego_state_se3_from_arrow_table(
            table,
            self._get_table_index(iteration),
            self.log_metadata.vehicle_parameters,
            timepoint=timepoint,
        )

    def get_box_detections_at_iteration(self, iteration: int) -> Optional[BoxDetectionWrapper]:
        """Inherited, see superclass."""
        table = self._get_modality_table(BOX_DETECTIONS_FILE)
        if table is None:
            return None
        timepoint = self.get_timepoint_at_iteration(iteration)
        return get_box_detections_se3_from_arrow_table(
            table,
            self._get_table_index(iteration),
            self.log_metadata,
            timepoint=timepoint,
        )

    def get_traffic_light_detections_at_iteration(self, iteration: int) -> Optional[TrafficLightDetectionWrapper]:
        """Inherited, see superclass."""
        table = self._get_modality_table(TRAFFIC_LIGHTS_FILE)
        if table is None:
            return None
        timepoint = self.get_timepoint_at_iteration(iteration)
        return get_traffic_light_detections_from_arrow_table(
            table, self._get_table_index(iteration), timepoint=timepoint
        )

    def get_route_lane_group_ids(self, iteration: int) -> Optional[List[int]]:
        """Inherited, see superclass."""
        table = self._get_modality_table(ROUTE_FILE)
        if table is None:
            return None
        return get_route_lane_group_ids_from_arrow_table(table, self._get_table_index(iteration))

    def get_pinhole_camera_at_iteration(
        self, iteration: int, camera_type: PinholeCameraType
    ) -> Optional[PinholeCamera]:
        """Inherited, see superclass."""
        pinhole_camera: Optional[PinholeCamera] = None
        if camera_type in self.available_pinhole_camera_types:
            camera_name = camera_type.serialize()
            table = self._get_modality_table(PINHOLE_CAMERA_FILE(camera_name))
            if table is not None:
                pinhole_camera_ = get_camera_from_arrow_table(
                    table,
                    self._get_table_index(iteration),
                    camera_type,
                    self.log_metadata,
                )
                assert isinstance(pinhole_camera_, PinholeCamera) or pinhole_camera_ is None
                pinhole_camera = pinhole_camera_
        return pinhole_camera

    def get_fisheye_mei_camera_at_iteration(
        self, iteration: int, camera_type: FisheyeMEICameraType
    ) -> Optional[FisheyeMEICamera]:
        """Inherited, see superclass."""
        fisheye_mei_camera: Optional[FisheyeMEICamera] = None
        if camera_type in self.available_fisheye_mei_camera_types:
            camera_name = camera_type.serialize()
            table = self._get_modality_table(FISHEYE_CAMERA_FILE(camera_name))
            if table is not None:
                fisheye_mei_camera_ = get_camera_from_arrow_table(
                    table,
                    self._get_table_index(iteration),
                    camera_type,
                    self.log_metadata,
                )
                assert isinstance(fisheye_mei_camera_, FisheyeMEICamera) or fisheye_mei_camera_ is None
                fisheye_mei_camera = fisheye_mei_camera_
        return fisheye_mei_camera

    def get_lidar_at_iteration(self, iteration: int, lidar_type: LiDARType) -> Optional[LiDAR]:
        """Inherited, see superclass."""
        lidar: Optional[LiDAR] = None
        if lidar_type in self.available_lidar_types or lidar_type == LiDARType.LIDAR_MERGED:
            lidar_name = lidar_type.serialize()
            table = self._get_modality_table(LIDAR_FILE(lidar_name))

            # If requesting LIDAR_MERGED but only individual files exist, merge them
            if table is None and lidar_type == LiDARType.LIDAR_MERGED:
                # Fall back to loading individual lidar files and merging
                import numpy as np

                from py123d.conversion.registry import DefaultLiDARIndex
                from py123d.datatypes.sensors import LiDARMetadata
                from py123d.geometry import PoseSE3

                point_clouds = []
                for lt in self.log_metadata.lidar_metadata.keys():
                    lt_table = self._get_modality_table(LIDAR_FILE(lt.serialize()))
                    if lt_table is not None:
                        lt_lidar = get_lidar_from_arrow_table(
                            lt_table, self._get_table_index(iteration), lt, self.log_metadata
                        )
                        if lt_lidar is not None:
                            point_clouds.append(lt_lidar.point_cloud)
                if point_clouds:
                    merged_pc = np.vstack(point_clouds)
                    lidar = LiDAR(
                        metadata=LiDARMetadata(
                            lidar_name=LiDARType.LIDAR_MERGED.serialize(),
                            lidar_type=LiDARType.LIDAR_MERGED,
                            lidar_index=DefaultLiDARIndex,
                            extrinsic=PoseSE3.identity(),
                        ),
                        point_cloud=merged_pc,
                    )
            elif table is not None:
                lidar = get_lidar_from_arrow_table(
                    table,
                    self._get_table_index(iteration),
                    lidar_type,
                    self.log_metadata,
                )
        return lidar
