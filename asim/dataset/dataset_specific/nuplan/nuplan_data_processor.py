import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Final, List, Tuple, Union

import pyarrow as pa
import pyarrow.ipc as ipc
import yaml
from nuplan.common.geometry.compute import get_pacifica_parameters
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
from tqdm import tqdm

import asim.dataset.dataset_specific.nuplan.utils as nuplan_utils
from asim.common.geometry.base import StateSE3
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3, BoundingBoxSE3Index
from asim.common.geometry.constants import DEFAULT_PITCH, DEFAULT_ROLL
from asim.common.geometry.vector import Vector3D
from asim.common.vehicle_state.ego_vehicle_state import DynamicVehicleState, EgoVehicleState, EgoVehicleStateIndex
from asim.dataset.dataset_specific.raw_data_processor import RawDataProcessor
from asim.dataset.logs.log_metadata import LogMetadata
from asim.dataset.observation.detection.detection import TrafficLightStatus
from asim.dataset.observation.detection.detection_types import DetectionType

NUPLAN_DT: Final[float] = 0.05
NUPLAN_FULL_MAP_NAME_DICT: Final[Dict[str, str]] = {
    "boston": "us-ma-boston",
    "singapore": "sg-one-north",
    "las_vegas": "us-nv-las-vegas-strip",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
}

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}
NUPLAN_DETECTION_NAME_DICT = {
    "vehicle": DetectionType.VEHICLE,
    "bicycle": DetectionType.BICYCLE,
    "pedestrian": DetectionType.PEDESTRIAN,
    "traffic_cone": DetectionType.TRAFFIC_CONE,
    "barrier": DetectionType.BARRIER,
    "czone_sign": DetectionType.CZONE_SIGN,
    "generic_object": DetectionType.GENERIC_OBJECT,
}

NUPLAN_DATA_ROOT = Path(os.environ["NUPLAN_DATA_ROOT"])


def create_splits_logs() -> Dict[str, List[str]]:
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r") as stream:
        splits = yaml.safe_load(stream)

    return splits["log_splits"]


class NuplanDataProcessor(RawDataProcessor):
    def __init__(self, splits: List[str], log_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        for split in splits:
            assert (
                split in self.get_available_splits()
            ), f"Split {split} is not available. Available splits: {self.available_splits}"

        self._splits: List[str] = splits
        self._log_path: Path = Path(log_path)
        self._output_path: Path = Path(output_path)
        self._log_paths_per_split: Dict[str, List[Path]] = self._collect_log_paths()

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        # NOTE: the nuplan mini folder has an internal train, val, test structure, all stored in "mini".
        # The complete dataset is saved in the "trainval" folder (train and val), or in the "test" folder (for test).
        subsplit_log_names: Dict[str, List[str]] = create_splits_logs()
        log_paths_per_split: Dict[str, List[Path]] = {}

        for split in self._splits:
            subsplit = split.split("_")[-1]
            assert subsplit in ["train", "val", "test"]
            if split in ["nuplan_train", "nuplan_val"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "trainval"
            elif split in ["nuplan_test"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "test"
            elif split in ["nuplan_mini_train", "nuplan_mini_val", "nuplan_mini_test"]:
                log_path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / "mini"

            all_log_files_in_path = [log_file for log_file in log_path.glob("*.db")]
            all_log_names = set([log_file.stem for log_file in all_log_files_in_path])
            split_log_names = set(subsplit_log_names[subsplit])
            log_paths = [log_path / f"{log_name}.db" for log_name in list(all_log_names & split_log_names)]
            log_paths_per_split[split] = log_paths

        return log_paths_per_split

    def get_available_splits(self) -> List[str]:
        return [
            "nuplan_train",
            "nuplan_val",
            "nuplan_test",
            "nuplan_mini_train",
            "nuplan_mini_val",
            "nuplan_mini_test",
        ]

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

        metadata_list = worker_map(worker, convert_nuplan_log_to_arrow, log_args)
        print(len(metadata_list))


def convert_nuplan_log_to_arrow(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
    def convert_log_internal(args: List[Dict[str, Union[List[str], List[Path]]]]) -> None:
        for log_info in args:
            log_path: Path = log_info["log_path"]
            output_path: Path = log_info["output_path"]
            split: str = log_info["split"]

            if not log_path.exists():
                raise FileNotFoundError(f"Log path {log_path} does not exist.")

            log_db = NuPlanDB(NUPLAN_DATA_ROOT, str(log_path), None)
            # tables["metadata_table"] = _get_metadata_table(log_db)
            recording_table = _get_recording_table(log_db)
            metadata = _get_metadata(log_db)
            recording_table = recording_table.replace_schema_metadata({"log_metadata": json.dumps(asdict(metadata))})

            log_file_path = output_path / split / f"{log_path.stem}.arrow"

            if not log_file_path.parent.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            with pa.OSFile(str(log_file_path), "wb") as sink:
                with ipc.new_file(sink, recording_table.schema) as writer:
                    writer.write_table(recording_table)

            del recording_table, log_db
            gc.collect()
        return []

    convert_log_internal(args)
    gc.collect()
    return []


def _get_metadata(log_db: NuPlanDB) -> LogMetadata:

    return LogMetadata(
        dataset="nuplan",
        log_name=log_db.log_name,
        location=log_db.log.map_version,
    )


def _get_recording_table(log_db: NuPlanDB) -> pa.Table:

    log_db.log.token
    log_db.log.map_version
    log_db.log.vehicle_name

    timestamp_log: List[int] = []

    detections_state_log: List[List[List[float]]] = []
    detections_token_log: List[List[str]] = []
    detections_type_log: List[List[int]] = []

    ego_states_log: List[List[float]] = []

    traffic_light_ids_log: List[List[int]] = []
    traffic_light_types_log: List[List[int]] = []
    scenario_tags_log: List[List[str]] = []

    for lidar_pc in tqdm(log_db.lidar_pc[::2], dynamic_ncols=True):
        lidar_pc_token: str = lidar_pc.token

        # 1. Timestamp (time_us)
        timestamp_log.append(lidar_pc.timestamp)

        # 2. box detections
        detections_state, detections_token, detections_types = _extract_detections(lidar_pc)
        detections_state_log.append(detections_state)
        detections_token_log.append(detections_token)
        detections_type_log.append(detections_types)

        # 3. Ego state
        ego_states_log.append(_extract_ego_state(lidar_pc))

        # 4. Traffic lights
        traffic_light_ids, traffic_light_types = _extract_traffic_lights(log_db, lidar_pc_token)
        traffic_light_ids_log.append(traffic_light_ids)
        traffic_light_types_log.append(traffic_light_types)

        # 5. Scenario Types
        scenario_tags_log.append(_extract_scenario_tag(log_db, lidar_pc_token))

    recording_data = {
        "timestamp": timestamp_log,
        "detections_state": detections_state_log,
        "detections_token": detections_token_log,
        "detections_type": detections_type_log,
        "ego_states": ego_states_log,
        "traffic_light_ids": traffic_light_ids_log,
        "traffic_light_types": traffic_light_types_log,
        "scenario_tag": scenario_tags_log,
    }

    # Create a PyArrow Table
    recording_schema = pa.schema(
        [
            ("timestamp", pa.int64()),
            ("detections_state", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
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


def _extract_detections(lidar_pc: LidarPc) -> Tuple[List[List[float]], List[str], List[int]]:
    detections_state: List[List[float]] = []
    detections_token: List[str] = []
    detections_types: List[int] = []

    for lidar_box in lidar_pc.lidar_boxes:
        lidar_box: LidarBox
        center = StateSE3(
            x=lidar_box.x,
            y=lidar_box.y,
            z=lidar_box.z,
            roll=DEFAULT_ROLL,
            pitch=DEFAULT_PITCH,
            yaw=lidar_box.yaw,
        )
        bounding_box_se3 = BoundingBoxSE3(center, lidar_box.length, lidar_box.width, lidar_box.height)

        detections_state.append(bounding_box_se3.array)
        detections_token.append(lidar_box.track_token)
        detections_types.append(int(NUPLAN_DETECTION_NAME_DICT[lidar_box.category.name]))

    return detections_state, detections_token, detections_types


def _extract_ego_state(lidar_pc: LidarPc) -> List[float]:

    yaw, pitch, roll = lidar_pc.ego_pose.quaternion.yaw_pitch_roll
    vehicle_parameters = get_pacifica_parameters()
    # TODO: Convert rear axle to center

    bounding_box = BoundingBoxSE3(
        center=StateSE3(
            x=lidar_pc.ego_pose.x,
            y=lidar_pc.ego_pose.y,
            z=lidar_pc.ego_pose.z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
        length=vehicle_parameters.length,
        width=vehicle_parameters.width,
        height=vehicle_parameters.height,
    )
    dynamic_state = DynamicVehicleState(
        velocity=Vector3D(
            x=lidar_pc.ego_pose.vx,
            y=lidar_pc.ego_pose.vy,
            z=lidar_pc.ego_pose.vz,
        ),
        acceleration=Vector3D(
            x=lidar_pc.ego_pose.acceleration_x,
            y=lidar_pc.ego_pose.acceleration_y,
            z=lidar_pc.ego_pose.acceleration_z,
        ),
        angular_velocity=Vector3D(
            x=lidar_pc.ego_pose.angular_rate_x,
            y=lidar_pc.ego_pose.angular_rate_y,
            z=lidar_pc.ego_pose.angular_rate_z,
        ),
    )

    return EgoVehicleState(bounding_box=bounding_box, dynamic_state=dynamic_state).array.tolist()


def _extract_traffic_lights(log_db: NuPlanDB, lidar_pc_token: str) -> Tuple[List[int], List[int]]:
    traffic_light_ids: List[int] = []
    traffic_light_types: List[int] = []
    traffic_lights = log_db.traffic_light_status.select_many(lidar_pc_token=lidar_pc_token)
    for traffic_light in traffic_lights:
        traffic_light_ids.append(int(traffic_light.lane_connector_id))
        traffic_light_types.append(int(NUPLAN_TRAFFIC_STATUS_DICT[traffic_light.status].value))
    return traffic_light_ids, traffic_light_types


def _extract_scenario_tag(log_db: NuPlanDB, lidar_pc_token: str) -> List[str]:

    scenario_tags = [
        scenario_tag.type for scenario_tag in log_db.scenario_tag.select_many(lidar_pc_token=lidar_pc_token)
    ]
    if len(scenario_tags) == 0:
        scenario_tags = ["unknown"]
    return scenario_tags
