import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pyarrow as pa

from asim.common.datatypes.detection.detection import BoxDetectionWrapper, TrafficLightDetectionWrapper
from asim.common.datatypes.time.time_point import TimePoint
from asim.common.datatypes.vehicle_state.ego_state import EgoStateSE3
from asim.common.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from asim.dataset.arrow.conversion import (
    get_box_detections_from_arrow_table,
    get_ego_vehicle_state_from_arrow_table,
    get_timepoint_from_arrow_table,
    get_traffic_light_detections_from_arrow_table,
)
from asim.dataset.arrow.helper import open_arrow_table
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.maps.gpkg.gpkg_map import get_map_api_from_names
from asim.dataset.scene.abstract_scene import AbstractScene, SceneExtractionInfo

# TODO: Remove or improve open/close dynamic of Scene object.


def _get_scene_data(arrow_file_path: Union[Path, str]) -> Tuple[LogMetadata, VehicleParameters]:
    """
    Extracts the metadata and vehicle parameters from the arrow file.
    """
    # TODO: consider a better way to read metadata, instead of loading the entire table.
    table = open_arrow_table(arrow_file_path)
    metadata = LogMetadata(**json.loads(table.schema.metadata[b"log_metadata"].decode()))
    vehicle_parameters = VehicleParameters(**json.loads(table.schema.metadata[b"vehicle_parameters"].decode()))
    del table
    return metadata, vehicle_parameters


class ArrowScene(AbstractScene):
    def __init__(
        self,
        arrow_file_path: Union[Path, str],
        scene_extraction_info: Optional[SceneExtractionInfo] = None,
    ) -> None:

        self._recording_table: pa.Table = None

        _metadata, _vehicle_parameters = _get_scene_data(arrow_file_path)
        self._metadata: LogMetadata = _metadata
        self._vehicle_parameters: VehicleParameters = _vehicle_parameters

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

    def get_ego_vehicle_state_at_iteration(self, iteration: int) -> EgoStateSE3:
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

    def get_route_lane_group_ids(self, iteration: int) -> List[int]:
        self._lazy_initialize()
        table_index = self._get_table_index(iteration)
        return self._recording_table["route_lane_group_ids"][table_index].as_py()

    def _lazy_initialize(self) -> None:
        self.open()

    def open(self) -> None:
        if self._map_api is None:
            self._map_api = get_map_api_from_names(self._metadata.dataset, self._metadata.location)
            self._map_api.initialize()
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
