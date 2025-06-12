import gc
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Final, List, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool
from tqdm import tqdm

from asim.common.geometry.base import Point3D
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from asim.common.geometry.vector import Vector3DIndex
from asim.common.vehicle_state.ego_vehicle_state import EgoVehicleStateIndex
from asim.dataset.dataset_specific.nuplan.nuplan_data_processor import worker_map
from asim.dataset.dataset_specific.raw_data_processor import RawDataProcessor
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.maps.abstract_map import AbstractMap, MapSurfaceType
from asim.dataset.maps.abstract_map_objects import AbstractLane
from asim.dataset.scene.arrow_scene import get_map_api_from_names

CARLA_DT: Final[float] = 0.1


TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE: Final[float] = 1.0  # [m]


def _load_json_gz(path: Path) -> Dict:
    """Helper function to load a gzipped JSON file."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data


class CarlaDataProcessor(RawDataProcessor):
    def __init__(self, splits: List[str], log_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: str = splits
        self._log_path: Path = Path(log_path)
        self._output_path: Path = Path(output_path)
        self._log_paths_per_split: Dict[str, List[Path]] = self._collect_log_paths()

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        # TODO: fix "carla" split placeholder and add support for other splits
        log_paths_per_split: Dict[str, List[Path]] = {}
        log_paths = list(self._log_path.iterdir())
        log_paths_per_split["carla"] = log_paths
        return log_paths_per_split

    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""
        return ["carla"]  # TODO: fix the placeholder

    def convert(self, worker: WorkerPool) -> None:

        log_args = [
            {
                "log_path": log_path,
                "output_path": self._output_path,
                "split": split,
            }
            for split, log_paths in self._log_paths_per_split.items()
            for log_path in log_paths
        ]

        worker_map(worker, convert_carla_log_to_arrow, log_args)


def convert_carla_log_to_arrow(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
    def convert_log_internal(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
        for log_info in args:
            log_path: Path = log_info["log_path"]
            output_path: Path = log_info["output_path"]
            split: str = log_info["split"]

            if not log_path.exists():
                raise FileNotFoundError(f"Log path {log_path} does not exist.")

            bounding_box_paths = sorted([bb_path for bb_path in (log_path / "boxes").iterdir()])
            map_name = _load_json_gz(bounding_box_paths[0])["location"]
            map_api = get_map_api_from_names("carla", map_name)

            recording_table = _get_recording_table(bounding_box_paths, map_api)
            metadata = _get_metadata(map_name, str(log_path.stem))
            recording_table = recording_table.replace_schema_metadata({"log_metadata": json.dumps(asdict(metadata))})
            log_file_path = output_path / split / f"{log_path.stem}.arrow"

            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            with pa.OSFile(str(log_file_path), "wb") as sink:
                with ipc.new_file(sink, recording_table.schema) as writer:
                    writer.write_table(recording_table)

            del recording_table
            gc.collect()

    convert_log_internal(args)
    gc.collect()
    return []


def _get_metadata(location: str, log_name: str) -> LogMetadata:
    return LogMetadata(dataset="carla", log_name=log_name, location=location, map_has_z=True)


def _get_recording_table(bounding_box_paths: List[Path], map_api: AbstractMap) -> pa.Table:

    timestamp_log: List[int] = []

    detections_state_log: List[List[List[float]]] = []
    detections_velocity_log: List[List[List[float]]] = []
    detections_token_log: List[List[str]] = []
    detections_type_log: List[List[int]] = []

    ego_states_log: List[List[float]] = []

    traffic_light_ids_log: List[List[int]] = []
    traffic_light_types_log: List[List[int]] = []
    scenario_tags_log: List[List[str]] = []

    for box_path in tqdm(bounding_box_paths):
        data = _load_json_gz(box_path)
        traffic_light_ids, traffic_light_types = _extract_traffic_light_data(
            data["traffic_light_states"], data["traffic_light_positions"], map_api
        )

        timestamp_log.append(data["timestamp"])
        detections_state_log.append(data["detections_state"])
        detections_velocity_log.append(data["detections_velocity"])
        detections_token_log.append(data["detections_token"])
        detections_type_log.append(data["detections_types"])
        ego_states_log.append(data["ego_state"])
        traffic_light_ids_log.append(traffic_light_ids)
        traffic_light_types_log.append(traffic_light_types)
        scenario_tags_log.append(data["scenario_tag"])

    recording_data = {
        "timestamp": timestamp_log,
        "detections_state": detections_state_log,
        "detections_velocity": detections_velocity_log,
        "detections_token": detections_token_log,
        "detections_type": detections_type_log,
        "ego_states": ego_states_log,
        "traffic_light_ids": traffic_light_ids_log,
        "traffic_light_types": traffic_light_types_log,
        "scenario_tag": scenario_tags_log,
    }

    recording_schema = pa.schema(
        [
            ("timestamp", pa.int64()),
            ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
            ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
            ("detections_token", pa.list_(pa.string())),
            ("detections_type", pa.list_(pa.int16())),
            ("ego_states", pa.list_(pa.float64(), len(EgoVehicleStateIndex))),
            ("traffic_light_ids", pa.list_(pa.int64())),
            ("traffic_light_types", pa.list_(pa.int16())),
            ("scenario_tag", pa.list_(pa.string())),
        ]
    )
    recording_table = pa.Table.from_pydict(recording_data, schema=recording_schema)
    recording_table = recording_table.sort_by([("timestamp", "ascending")])

    return recording_table


def _extract_traffic_light_data(
    traffic_light_states: List[int], traffic_light_positions: List[List[float]], map_api: AbstractMap
) -> Tuple[List[int], List[int]]:
    traffic_light_types: List[int] = []
    traffic_light_ids: List[int] = []
    for traffic_light_state, traffic_light_waypoints in zip(traffic_light_states, traffic_light_positions):
        for traffic_light_waypoint in traffic_light_waypoints:
            point_3d = Point3D(*traffic_light_waypoint)
            nearby_lanes = map_api.get_proximal_map_objects(
                point_3d, TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE, [MapSurfaceType.LANE]
            )[MapSurfaceType.LANE]

            for lane in nearby_lanes:
                lane: AbstractLane
                lane_start_point = lane.centerline.array[0]
                distance_to_lane_start = np.linalg.norm(lane_start_point - point_3d.array)
                if distance_to_lane_start < TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE:
                    traffic_light_ids.append(int(lane.id))
                    traffic_light_types.append(traffic_light_state)
    return traffic_light_ids, traffic_light_types
