import gc
import gzip
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Dict, Final, List, Tuple, Union

import numpy as np
import pyarrow as pa
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
from traitlets import Any

from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import get_carla_lincoln_mkz_2020_parameters
from d123.common.geometry.base import Point2D, Point3D, StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from d123.common.geometry.transform.se3 import translate_se3_along_z
from d123.common.geometry.vector import Vector3DIndex
from d123.dataset.arrow.conversion import VehicleParameters
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.dataset_specific.carla.opendrive.elements.opendrive import OpenDrive
from d123.dataset.dataset_specific.carla.opendrive.opendrive_converter import OpenDriveConverter
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata
from d123.dataset.maps.abstract_map import AbstractMap, MapSurfaceType
from d123.dataset.maps.abstract_map_objects import AbstractLane
from d123.dataset.scene.arrow_scene import get_map_api_from_names

AVAILABLE_CARLA_MAP_LOCATIONS: Final[List[str]] = [
    "Town01",  # A small, simple town with a river and several bridges.
    "Town02",  # A small simple town with a mixture of residential and commercial buildings.
    "Town03",  # A larger, urban map with a roundabout and large junctions.
    "Town04",  # A small town embedded in the mountains with a special "figure of 8" infinite highway.
    "Town05",  # Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
    "Town06",  # Long many lane highways with many highway entrances and exits. It also has a Michigan left.
    "Town07",  # A rural environment with narrow roads, corn, barns and hardly any traffic lights.
    "Town10HD",  # A downtown urban environment with skyscrapers, residential buildings and an ocean promenade.
    "Town11",  # A Large Map that is undecorated. Serves as a proof of concept for the Large Maps feature.
    "Town12",  # A Large Map with numerous different regions, including high-rise, residential and rural environments.
    "Town13",  # ???
    "Town15",  # ???
]

CARLA_DT: Final[float] = 0.1  # [s]
TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE: Final[float] = 1.0  # [m]
SORT_BY_TIMESTAMP: Final[bool] = True


# TODO: Refactor this files and convert coordinate systems more elegantly.
# NOTE: Currently some coordinate transforms from Unreal to ISO 8855 are done in the data agent of PDM-Lite.
# Ideally a general function to transform poses and points between coordinate systems would be nice


def _load_json_gz(path: Path) -> Dict:
    """Helper function to load a gzipped JSON file."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data


def create_token(input_data: str) -> str:
    # TODO: Refactor this function.
    # TODO: Add a general function to create tokens from arbitrary data.
    if isinstance(input_data, str):
        input_data = input_data.encode("utf-8")

    hash_obj = hashlib.sha256(input_data)
    return hash_obj.hexdigest()[:16]


class CarlaDataConverter(RawDataConverter):

    def __init__(
        self,
        splits: List[str],
        log_path: Union[Path, str],
        data_converter_config: DataConverterConfig,
    ) -> None:
        super().__init__(data_converter_config)
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: str = splits
        self._log_path: Path = Path(log_path)
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

    def convert_maps(self, worker: WorkerPool) -> None:
        worker_map(
            worker,
            partial(
                convert_carla_map_to_gpkg,
                output_path=self._output_path,
                force_map_conversion=self.force_map_conversion,
            ),
            list(AVAILABLE_CARLA_MAP_LOCATIONS),
        )

    def convert_logs(self, worker: WorkerPool) -> None:

        log_args = [
            {
                "log_path": log_path,
                "output_path": self._output_path,
                "split": split,
            }
            for split, log_paths in self._log_paths_per_split.items()
            for log_path in log_paths
        ]

        worker_map(
            worker, partial(convert_carla_log_to_arrow, force_log_conversion=self.force_log_conversion), log_args
        )


def convert_carla_map_to_gpkg(map_names: List[str], output_path: Path, force_map_conversion: bool) -> List[Any]:
    for map_name in map_names:
        map_path = output_path / "maps" / f"carla_{map_name}.gpkg"
        if force_map_conversion or not map_path.exists():
            map_path.unlink(missing_ok=True)
            assert os.environ["CARLA_ROOT"] is not None
            CARLA_ROOT = Path(os.environ["CARLA_ROOT"])

            if map_name not in ["Town11", "Town12", "Town13", "Town15"]:
                carla_maps_root = CARLA_ROOT / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive"
                carla_map_path = carla_maps_root / f"{map_name}.xodr"
            else:
                carla_map_path = (
                    CARLA_ROOT / "CarlaUE4" / "Content" / "Carla" / "Maps" / map_name / "OpenDrive" / f"{map_name}.xodr"
                )

            OpenDriveConverter(OpenDrive.parse_from_file(carla_map_path)).run(f"carla_{map_name.lower()}")

    return []


def convert_carla_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]],
    force_log_conversion: bool,
) -> List[Any]:
    def convert_log_internal(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
        for log_info in args:
            log_path: Path = log_info["log_path"]
            output_path: Path = log_info["output_path"]
            split: str = log_info["split"]

            log_file_path = output_path / split / f"{log_path.stem}.arrow"

            if force_log_conversion or not log_file_path.exists():
                log_file_path.unlink(missing_ok=True)
                if not log_file_path.parent.exists():
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)

                bounding_box_paths = sorted([bb_path for bb_path in (log_path / "boxes").iterdir()])
                map_name = _load_json_gz(bounding_box_paths[0])["location"]
                map_api = get_map_api_from_names("carla", map_name)

                recording_schema = pa.schema(
                    [
                        ("token", pa.string()),
                        ("timestamp", pa.int64()),
                        ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                        ("detections_velocity", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                        ("detections_token", pa.list_(pa.string())),
                        ("detections_type", pa.list_(pa.int16())),
                        ("ego_states", pa.list_(pa.float64(), len(EgoStateSE3Index))),
                        ("traffic_light_ids", pa.list_(pa.int64())),
                        ("traffic_light_types", pa.list_(pa.int16())),
                        ("scenario_tag", pa.list_(pa.string())),
                        ("route_lane_group_ids", pa.list_(pa.int64())),
                        ("front_cam_demo", pa.binary()),
                        ("front_cam_transform", pa.list_(pa.float64())),
                    ]
                )
                metadata = _get_metadata(map_name, str(log_path.stem))
                vehicle_parameters = _get_vehicle_parameters_()
                recording_schema = recording_schema.with_metadata(
                    {
                        "log_metadata": json.dumps(asdict(metadata)),
                        "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                    }
                )

                _write_recording_table(bounding_box_paths, map_api, recording_schema, log_file_path)

            gc.collect()

    convert_log_internal(args)
    gc.collect()
    return []


def _get_metadata(location: str, log_name: str) -> LogMetadata:
    return LogMetadata(
        dataset="carla",
        log_name=log_name,
        location=location,
        timestep_seconds=CARLA_DT,
        map_has_z=True,
    )


def _get_vehicle_parameters(vehicle_parameter_dict: Dict[str, float]) -> VehicleParameters:
    # NOTE: @DanielDauner extracting the vehicle parameters from CARLA is somewhat tricky.
    # Need to extract the coordinates (for wheels, wheelbase, etc.) from CARLA which is somewhat noise.
    # Thus, we hardcode the parameters for the Lincoln MKZ 2020 (default).
    assert (
        vehicle_parameter_dict["vehicle_name"] == "vehicle.lincoln.mkz_2020"
    ), "Currently only supports MKZ 2020 in CARLA."
    return get_carla_lincoln_mkz_2020_parameters()


def _get_vehicle_parameters_() -> VehicleParameters:
    return get_carla_lincoln_mkz_2020_parameters()


def _write_recording_table(
    bounding_box_paths: List[Path],
    map_api: AbstractMap,
    recording_schema: pa.Schema,
    log_file_path: Path,
) -> pa.Table:
    log_path = bounding_box_paths[0].parent.parent.stem

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            for box_path in bounding_box_paths:
                data = _load_json_gz(box_path)
                traffic_light_ids, traffic_light_types = _extract_traffic_light_data(
                    data["traffic_light_states"], data["traffic_light_positions"], map_api
                )
                route_lane_group_ids = _extract_route_lane_group_ids(data["route"], map_api) if "route" in data else []
                front_cam_demo, front_cam_transform = _extract_front_cam_demo(box_path)

                row_data = {
                    "token": [create_token(f"{str(log_path)}_{box_path.stem}")],
                    "timestamp": [data["timestamp"]],
                    "detections_state": [_extract_detection_states(data["detections_state"])],
                    "detections_velocity": [
                        (
                            data["detections_velocity"]
                            if "detections_velocity" in data
                            else np.zeros((len(data["detections_types"]), 3)).tolist()
                        )
                    ],
                    "detections_token": [data["detections_token"]],
                    "detections_type": [data["detections_types"]],
                    "ego_states": [_extract_ego_vehicle_state(data["ego_state"])],
                    "traffic_light_ids": [traffic_light_ids],
                    "traffic_light_types": [traffic_light_types],
                    "scenario_tag": [data["scenario_tag"]],
                    "route_lane_group_ids": [route_lane_group_ids],
                    "front_cam_demo": [front_cam_demo],
                    "front_cam_transform": [front_cam_transform],
                }
                batch = pa.record_batch(row_data, schema=recording_schema)
                writer.write_batch(batch)
                del batch, row_data

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)


def _extract_ego_vehicle_state(ego_state_list: List[float]) -> List[float]:

    # NOTE: @DanielDauner This function is a temporary workaround.
    # CARLAs bounding box location starts at bottom vertically.
    # Need to translate half the height along the z-axis.
    ego_state_array = np.array(ego_state_list, dtype=np.float64)
    vehicle_parameters = get_carla_lincoln_mkz_2020_parameters()
    center = StateSE3.from_array(ego_state_array[EgoStateSE3Index.SE3])
    center = translate_se3_along_z(center, vehicle_parameters.height / 2)
    ego_state_array[EgoStateSE3Index.SE3] = center.array

    return ego_state_array.tolist()


def _extract_detection_states(detection_states: List[List[float]]) -> List[float]:

    # NOTE: @DanielDauner This function is a temporary workaround.
    # CARLAs bounding box location starts at bottom vertically.
    # Need to translate half the height along the z-axis.

    detection_state_converted = []

    for detection_state in detection_states:
        detection_state_array = np.array(detection_state, dtype=np.float64)
        center = StateSE3.from_array(detection_state_array[BoundingBoxSE3Index.STATE_SE3])
        center = translate_se3_along_z(center, detection_state_array[BoundingBoxSE3Index.HEIGHT] / 2)
        detection_state_array[EgoStateSE3Index.SE3] = center.array
        detection_state_converted.append(detection_state_array.tolist())

    return detection_state_converted


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


def _extract_route_lane_group_ids(route: List[List[float]], map_api: AbstractMap) -> List[int]:

    route = np.array(route, dtype=np.float64)
    route[..., 1] = -route[..., 1]  # Unreal coordinate system to ISO 8855
    route = route[::2]

    route_lane_group_ids: List[int] = []

    for point in route[:200]:
        point_2d = Point2D(point[0], point[1])
        nearby_lane_groups = map_api.query(point_2d.shapely_point, [MapSurfaceType.LANE_GROUP], predicate="intersects")[
            MapSurfaceType.LANE_GROUP
        ]
        if len(nearby_lane_groups) == 0:
            continue
        elif len(nearby_lane_groups) > 1:
            possible_lane_group_ids = [lane_group.id for lane_group in nearby_lane_groups]
            if len(route_lane_group_ids) > 0:
                prev_lane_group_id = route_lane_group_ids[-1]
                if prev_lane_group_id in possible_lane_group_ids:
                    continue
                else:
                    # TODO: Choose with least heading difference?
                    route_lane_group_ids.append(int(nearby_lane_groups[0].id))
            else:
                # TODO: Choose with least heading difference?
                route_lane_group_ids.append(int(nearby_lane_groups[0].id))
        elif len(nearby_lane_groups) == 1:
            route_lane_group_ids.append(int(nearby_lane_groups[0].id))

    return list(dict.fromkeys(route_lane_group_ids))  # Remove duplicates while preserving order


def _extract_front_cam_demo(box_path: Path) -> Tuple[bytes, List[float]]:

    sample_name = str(box_path.stem).split(".")[0]
    sensor_root = Path(box_path.parent.parent) / "rgb"

    front_cam_demo: bytes = None
    front_cam_transform: List[float] = []

    jpg_path = sensor_root / f"{sample_name}.jpg"

    if jpg_path.exists():
        with open(jpg_path, "rb") as f:
            front_cam_demo = f.read()

    return front_cam_demo, front_cam_transform
