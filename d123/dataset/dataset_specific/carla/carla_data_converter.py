import gc
import gzip
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar import LiDARMetadata, LiDARType, lidar_metadata_dict_to_json
from d123.common.datatypes.sensor.lidar_index import CarlaLidarIndex
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE3Index
from d123.common.datatypes.vehicle_state.vehicle_parameters import get_carla_lincoln_mkz_2020_parameters
from d123.common.multithreading.worker_utils import WorkerPool, worker_map
from d123.dataset.arrow.helper import open_arrow_table, write_arrow_table
from d123.dataset.conversion.map.opendrive.opendrive_map_conversion import convert_from_xodr
from d123.dataset.dataset_specific.raw_data_converter import DataConverterConfig, RawDataConverter
from d123.dataset.logs.log_metadata import LogMetadata
from d123.dataset.maps.abstract_map import AbstractMap, MapLayer
from d123.dataset.maps.abstract_map_objects import AbstractLane
from d123.dataset.scene.arrow_scene import get_map_api_from_names
from d123.geometry import BoundingBoxSE3Index, Point2D, Point3D, Vector3DIndex

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

CARLA_CAMERA_TYPES = {CameraType.CAM_F0}

CARLA_DATA_ROOT: Final[Path] = Path(os.environ["CARLA_DATA_ROOT"])


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
        log_paths = [
            log_path for log_path in self._log_path.iterdir() if log_path.is_dir() and log_path.stem != "sensor_blobs"
        ]
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
                data_converter_config=self.data_converter_config,
            ),
            list(AVAILABLE_CARLA_MAP_LOCATIONS),
        )

    def convert_logs(self, worker: WorkerPool) -> None:

        log_args = [
            {"log_path": log_path, "split": split}
            for split, log_paths in self._log_paths_per_split.items()
            for log_path in log_paths
        ]

        worker_map(
            worker, partial(convert_carla_log_to_arrow, data_converter_config=self.data_converter_config), log_args
        )


def convert_carla_map_to_gpkg(map_names: List[str], data_converter_config: DataConverterConfig) -> List[Any]:

    # TODO: add to config
    _interpolation_step_size = 0.5  # [m]
    _connection_distance_threshold = 0.1  # [m]
    for map_name in map_names:
        map_path = data_converter_config.output_path / "maps" / f"carla_{map_name.lower()}.gpkg"
        if data_converter_config.force_map_conversion or not map_path.exists():
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

            convert_from_xodr(
                carla_map_path,
                f"carla_{map_name.lower()}",
                _interpolation_step_size,
                _connection_distance_threshold,
            )
    return []


def convert_carla_log_to_arrow(
    args: List[Dict[str, Union[List[str], List[Path]]]], data_converter_config: DataConverterConfig
) -> List[Any]:
    def convert_log_internal(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
        for log_info in args:
            log_path: Path = log_info["log_path"]
            split: str = log_info["split"]
            output_path: Path = data_converter_config.output_path

            log_file_path = output_path / split / f"{log_path.stem}.arrow"

            if data_converter_config.force_log_conversion or not log_file_path.exists():
                log_file_path.unlink(missing_ok=True)
                if not log_file_path.parent.exists():
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)

                bounding_box_paths = sorted([bb_path for bb_path in (log_path / "boxes").iterdir()])
                first_log_dict = _load_json_gz(bounding_box_paths[0])
                map_name = first_log_dict["location"]
                map_api = get_map_api_from_names("carla", map_name)

                metadata = _get_metadata(map_name, str(log_path.stem))
                vehicle_parameters = get_carla_lincoln_mkz_2020_parameters()
                camera_metadata = get_carla_camera_metadata(first_log_dict)
                lidar_metadata = get_carla_lidar_metadata(first_log_dict)

                schema_column_list = [
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
                ]
                if data_converter_config.lidar_store_option is not None:
                    for lidar_type in lidar_metadata.keys():
                        if data_converter_config.lidar_store_option == "path":
                            schema_column_list.append((lidar_type.serialize(), pa.string()))
                        elif data_converter_config.lidar_store_option == "binary":
                            raise NotImplementedError("Binary lidar storage is not implemented.")

                # TODO: Adjust how cameras are added
                if data_converter_config.camera_store_option is not None:
                    for camera_type in camera_metadata.keys():
                        if data_converter_config.camera_store_option == "path":
                            schema_column_list.append((camera_type.serialize(), pa.string()))
                            schema_column_list.append(
                                (f"{camera_type.serialize()}_extrinsic", pa.list_(pa.float64(), 16))
                            )
                        elif data_converter_config.camera_store_option == "binary":
                            raise NotImplementedError("Binary camera storage is not implemented.")

                recording_schema = pa.schema(schema_column_list)
                recording_schema = recording_schema.with_metadata(
                    {
                        "log_metadata": json.dumps(asdict(metadata)),
                        "vehicle_parameters": json.dumps(asdict(vehicle_parameters)),
                        "camera_metadata": camera_metadata_dict_to_json(camera_metadata),
                        "lidar_metadata": lidar_metadata_dict_to_json(lidar_metadata),
                    }
                )

                _write_recording_table(
                    bounding_box_paths,
                    map_api,
                    recording_schema,
                    log_file_path,
                    data_converter_config,
                )

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


def get_carla_camera_metadata(first_log_dict: Dict[str, Any]) -> Dict[CameraType, CameraMetadata]:

    # FIXME: This is a placeholder function to return camera metadata.

    intrinsic = np.array(
        first_log_dict[f"{CameraType.CAM_F0.serialize()}_intrinsics"],
        dtype=np.float64,
    )
    camera_metadata = {
        CameraType.CAM_F0: CameraMetadata(
            camera_type=CameraType.CAM_F0,
            width=1024,
            height=512,
            intrinsic=intrinsic,
            distortion=np.zeros((5,), dtype=np.float64),
        )
    }
    return camera_metadata


def get_carla_lidar_metadata(first_log_dict: Dict[str, Any]) -> Dict[LiDARType, LiDARMetadata]:

    # TODO: add lidar extrinsic
    lidar_metadata = {
        LiDARType.LIDAR_TOP: LiDARMetadata(
            lidar_type=LiDARType.LIDAR_TOP,
            lidar_index=CarlaLidarIndex,
            extrinsic=None,
        )
    }
    return lidar_metadata


def _write_recording_table(
    bounding_box_paths: List[Path],
    map_api: AbstractMap,
    recording_schema: pa.Schema,
    log_file_path: Path,
    data_converter_config: DataConverterConfig,
) -> pa.Table:
    # TODO: Refactor this function to be more readable
    log_name = str(bounding_box_paths[0].parent.parent.stem)

    with pa.OSFile(str(log_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, recording_schema) as writer:
            for box_path in bounding_box_paths:
                sample_name = box_path.stem.split(".")[0]

                data = _load_json_gz(box_path)
                traffic_light_ids, traffic_light_types = _extract_traffic_light_data(
                    data["traffic_light_states"], data["traffic_light_positions"], map_api
                )
                route_lane_group_ids = _extract_route_lane_group_ids(data["route"], map_api) if "route" in data else []

                row_data = {
                    "token": [create_token(f"{log_name}_{box_path.stem}")],
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
                }
                if data_converter_config.lidar_store_option is not None:
                    lidar_data_dict = _extract_lidar(log_name, sample_name, data_converter_config)
                    for lidar_type, lidar_data in lidar_data_dict.items():
                        row_data[lidar_type.serialize()] = [lidar_data]

                if data_converter_config.camera_store_option is not None:
                    camera_data_dict = _extract_cameras(data, log_name, sample_name, data_converter_config)
                    for camera_type, camera_data in camera_data_dict.items():
                        if camera_data is not None:
                            row_data[camera_type.serialize()] = [camera_data[0]]
                            row_data[f"{camera_type.serialize()}_extrinsic"] = [camera_data[1]]
                        else:
                            row_data[camera_type.serialize()] = [None]

                batch = pa.record_batch(row_data, schema=recording_schema)
                writer.write_batch(batch)
                del batch, row_data

    if SORT_BY_TIMESTAMP:
        recording_table = open_arrow_table(log_file_path)
        recording_table = recording_table.sort_by([("timestamp", "ascending")])
        write_arrow_table(recording_table, log_file_path)


def _extract_ego_vehicle_state(ego_state_list: List[float]) -> List[float]:
    # NOTE: This function used to convert coordinate systems, but it is not needed anymore.
    assert len(ego_state_list) == len(EgoStateSE3Index), "Ego state list has incorrect length."
    return ego_state_list


def _extract_detection_states(detection_states: List[List[float]]) -> List[List[float]]:
    # NOTE: This function used to convert coordinate systems, but it is not needed anymore.
    detection_state_converted = []
    for detection_state in detection_states:
        assert len(detection_state) == len(BoundingBoxSE3Index), "Detection state has incorrect length."
        detection_state_converted.append(detection_state)
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
                point_3d, TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE, [MapLayer.LANE]
            )[MapLayer.LANE]

            for lane in nearby_lanes:
                lane: AbstractLane
                lane_start_point = lane.centerline.array[0]
                distance_to_lane_start = np.linalg.norm(lane_start_point - point_3d.array)
                if distance_to_lane_start < TRAFFIC_LIGHT_ASSIGNMENT_DISTANCE:
                    traffic_light_ids.append(int(lane.id))
                    traffic_light_types.append(traffic_light_state)
    return traffic_light_ids, traffic_light_types


def _extract_route_lane_group_ids(route: List[List[float]], map_api: AbstractMap) -> List[int]:

    # FIXME: Carla route is very buggy. No check if lanes are connected.
    route = np.array(route, dtype=np.float64)
    route[..., 1] = -route[..., 1]  # Unreal coordinate system to ISO 8855
    route = route[::2]

    route_lane_group_ids: List[int] = []

    for point in route[:200]:
        point_2d = Point2D(point[0], point[1])
        nearby_lane_groups = map_api.query(point_2d.shapely_point, [MapLayer.LANE_GROUP], predicate="intersects")[
            MapLayer.LANE_GROUP
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


def _extract_cameras(
    data: Dict[str, Any], log_name: str, sample_name: str, data_converter_config: DataConverterConfig
) -> Dict[CameraType, Optional[str]]:
    camera_dict: Dict[str, Union[str, bytes]] = {}
    for camera_type in CARLA_CAMERA_TYPES:
        camera_full_path = CARLA_DATA_ROOT / "sensor_blobs" / log_name / camera_type.name / f"{sample_name}.jpg"
        if camera_full_path.exists():
            if data_converter_config.camera_store_option == "path":
                path = f"{log_name}/{camera_type.name}/{sample_name}.jpg"
                extrinsics = data.get(f"{camera_type.serialize()}_transform", None)
                camera_dict[camera_type] = path, (
                    np.array(extrinsics, dtype=np.float64).flatten() if extrinsics is not None else None
                )

            elif data_converter_config.camera_store_option == "binary":
                raise NotImplementedError("Binary camera storage is not implemented.")
        else:
            camera_dict[camera_type] = None
    return camera_dict


def _extract_lidar(
    log_name: str, sample_name: str, data_converter_config: DataConverterConfig
) -> Dict[LiDARType, Optional[str]]:

    lidar: Optional[str] = None
    lidar_full_path = CARLA_DATA_ROOT / "sensor_blobs" / log_name / "lidar" / f"{sample_name}.npy"
    if lidar_full_path.exists():
        if data_converter_config.lidar_store_option == "path":
            lidar = f"{log_name}/lidar/{sample_name}.npy"
        elif data_converter_config.lidar_store_option == "binary":
            raise NotImplementedError("Binary lidar storage is not implemented.")
    else:
        raise FileNotFoundError(f"LiDAR file not found: {lidar_full_path}")

    return {LiDARType.LIDAR_TOP: lidar} if lidar else None
