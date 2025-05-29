import os
from pathlib import Path
from typing import Dict, Final, List

import pyarrow as pa
from nuplan.common.geometry.compute import get_pacifica_parameters
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from tqdm import tqdm

from asim.common.geometry.base import StateSE3
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from asim.common.geometry.constants import DEFAULT_PITCH, DEFAULT_ROLL
from asim.dataset.observation.agent_datatypes import BoundingBoxType
from asim.dataset.observation.traffic_light import TrafficLightStatusType

NUPLAN_DT: Final[float] = 0.05
NUPLAN_FULL_MAP_NAME_DICT: Final[Dict[str, str]] = {
    "boston": "us-ma-boston",
    "singapore": "sg-one-north",
    "las_vegas": "us-nv-las-vegas-strip",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
}
_NUPLAN_SQL_MAP_FRIENDLY_NAMES_DICT: Final[Dict[str, str]] = {
    "us-ma-boston": "boston",
    "sg-one-north": "singapore",
    "las_vegas": "las_vegas",
    "us-pa-pittsburgh-hazelwood": "pittsburgh",
}

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatusType]] = {
    "green": TrafficLightStatusType.GREEN,
    "red": TrafficLightStatusType.RED,
    "unknown": TrafficLightStatusType.UNKNOWN,
}
name_mapping = {
    "vehicle": BoundingBoxType.VEHICLE,
    "bicycle": BoundingBoxType.BICYCLE,
    "pedestrian": BoundingBoxType.PEDESTRIAN,
    "traffic_cone": BoundingBoxType.TRAFFIC_CONE,
    "barrier": BoundingBoxType.BARRIER,
    "czone_sign": BoundingBoxType.CZONE_SIGN,
    "generic_object": BoundingBoxType.GENERIC_OBJECT,
}

NUPLAN_DATA_ROOT = Path(os.environ["NUPLAN_DATA_ROOT"])


class NuPlanDataset:
    def __init__(self, output_path: Path, split: str) -> None:
        assert split in [
            "train",
            "val",
            "test",
            "mini",
        ], f"Invalid split: {split}. Must be one of ['train', 'val', 'test', 'mini']."
        self._log_path: Path = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "splits" / split

    def convert(self, log_name: str) -> None:

        log_path = self._log_path / f"{log_name}.db"
        if not log_path.exists():
            raise FileNotFoundError(f"Log path {log_path} does not exist.")

        NuPlanDB(NUPLAN_DATA_ROOT, str(log_path), None)

        metadata = {
            "recording_id": "drive_20250515_001",
            "location": "Mountain View, CA",
            "weather": "sunny",
            "sensor_config": "standard_suite_v3",
        }
        metadata_fields = []
        metadata_values = []
        for key, value in metadata.items():
            metadata_fields.append(key)
            metadata_values.append(pa.scalar(value))

        pa.Table.from_arrays([pa.array([value]) for value in metadata_values], metadata_fields)

    def _get_frame_table(self, log_db: NuPlanDB) -> pa.Table:

        log_db.log.token
        log_db.log.map_version
        log_db.log.vehicle_name

        time_us_log: List[int] = []

        bb_ego_log: List[List[float]] = []
        bb_frame_log: List[List[List[float]]] = []
        bb_track_log: List[List[str]] = []
        bb_types_log: List[List[int]] = []

        ego_states_log: List[List[float]] = []

        for lidar_pc in tqdm(log_db.lidar_pc, dynamic_ncols=True):

            # 1. time_us
            time_us_log.append(lidar_pc.timestamp)

            # 2. Non-ego bounding boxes
            bb_frame: List[List[float]] = []
            bb_track: List[str] = []
            bb_types: List[int] = []

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

                bb_frame.append(pa.array(bounding_box_se3.array))
                bb_track.append(lidar_box.track_token)
                bb_types.append(int(name_mapping[lidar_box.category.name]))

            bb_frame_log.append(bb_frame)
            bb_track_log.append(bb_track)
            bb_types_log.append(bb_types)

            # 3. ego_states
            yaw, pitch, roll = lidar_pc.ego_pose.quaternion.yaw_pitch_roll
            vehicle_parameters = get_pacifica_parameters()
            # TODO: Convert rear axle to center

            ego_bounding_box_se3 = BoundingBoxSE3(
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

            bb_ego_log.append(pa.array(ego_bounding_box_se3.array))

        frame_data = {
            "time_us": time_us_log,
            "bb_frame": bb_frame_log,
            "bb_track": bb_track_log,
            "bb_types": bb_types_log,
            "bb_ego": bb_ego_log,
        }

        # Create a PyArrow Table
        list_schema = pa.schema(
            [
                ("time_us", pa.int64()),
                ("bb_frame", pa.list_(pa.list_(pa.float64(), 9))),
                ("bb_track", pa.list_(pa.string())),
                ("bb_types", pa.list_(pa.int32())),
                ("bb_ego", pa.list_(pa.float64(), 9)),
            ]
        )
        list_table = pa.Table.from_pydict(frame_data, schema=list_schema)
        return list_table
