import glob
import sqlite3
from pathlib import Path
from typing import Dict, Final, Generator, Iterable, List, Optional

import pandas as pd

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


class NuPlanDataset:
    def __init__(self, dataset_path: Path, subfolder: str) -> None:
        self.base_path: Path = dataset_path / subfolder

        self.connection: sqlite3.Connection = None
        self.cursor: sqlite3.Cursor = None

        self.scenes: List[Dict[str, str]] = self._load_scenes()

    def open_db(self, db_filename: str) -> None:
        self.connection = sqlite3.connect(str(self.base_path / db_filename))
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

    def execute_query_one(self, query_text: str, query_params: Optional[Iterable] = None) -> sqlite3.Row:
        self.cursor.execute(query_text, query_params if query_params is not None else [])
        return self.cursor.fetchone()

    def execute_query_all(self, query_text: str, query_params: Optional[Iterable] = None) -> List[sqlite3.Row]:
        self.cursor.execute(query_text, query_params if query_params is not None else [])
        return self.cursor.fetchall()

    def execute_query_iter(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> Generator[sqlite3.Row, None, None]:
        self.cursor.execute(query_text, query_params if query_params is not None else [])

        for row in self.cursor:
            yield row

    def _load_scenes(self) -> List[Dict[str, str]]:
        scene_info_query = """
        SELECT  sc.token AS scene_token,
                log.location,
                log.logfile,
                (
                    SELECT count(*)
                    FROM lidar_pc AS lpc
                    WHERE lpc.scene_token = sc.token
                ) AS num_timesteps
        FROM scene AS sc
        LEFT JOIN log ON sc.log_token = log.token
        """
        scenes: List[Dict[str, str]] = []

        for log_filename in glob.glob(str(self.base_path / "*.db")):
            self.open_db(log_filename)

            for row in self.execute_query_iter(scene_info_query):
                scenes.append(
                    {
                        "name": f"{row['logfile']}={row['scene_token'].hex()}",
                        "location": _NUPLAN_SQL_MAP_FRIENDLY_NAMES_DICT[row["location"]],
                        "num_timesteps": row["num_timesteps"],
                    }
                )

            self.close_db()

        return scenes

    def get_scene_frames(self, scene_token_str: str) -> pd.DataFrame:
        query = """
        SELECT  lpc.token AS lpc_token,
                ep.x AS ego_x,
                ep.y AS ego_y,
                ep.z AS ego_z,
                ep.qw AS ego_qw,
                ep.qx AS ego_qx,
                ep.qy AS ego_qy,
                ep.qz AS ego_qz,
                ep.vx AS ego_vx,
                ep.vy AS ego_vy,
                ep.acceleration_x AS ego_ax,
                ep.acceleration_y AS ego_ay
        FROM lidar_pc AS lpc
        LEFT JOIN ego_pose AS ep ON lpc.ego_pose_token = ep.token
        WHERE scene_token = ?
        ORDER BY lpc.timestamp ASC;
        """
        # log_filename, scene_token_str = scene.name.split("=")
        scene_token = bytearray.fromhex(scene_token_str)

        return pd.read_sql_query(query, self.connection, index_col="lpc_token", params=(scene_token,))

    def get_detected_agents(self, binary_lpc_tokens: List[bytearray]) -> pd.DataFrame:
        query = f"""
        SELECT  lb.lidar_pc_token,
                lb.track_token,
                (SELECT category.name FROM category WHERE category.token = tr.category_token) AS category_name,
                tr.width,
                tr.length,
                tr.height,
                lb.x,
                lb.y,
                lb.z,
                lb.vx,
                lb.vy,
                lb.yaw
        FROM lidar_box AS lb
        LEFT JOIN track AS tr ON lb.track_token = tr.token

        WHERE lidar_pc_token IN ({('?,'*len(binary_lpc_tokens))[:-1]}) AND category_name IN ('vehicle', 'bicycle', 'pedestrian')
        """
        return pd.read_sql_query(query, self.connection, params=binary_lpc_tokens)

    def get_traffic_light_status(self, binary_lpc_tokens: List[bytearray]) -> pd.DataFrame:
        query = f"""
        SELECT  tls.lidar_pc_token AS lidar_pc_token,
                tls.lane_connector_id AS lane_id,
                tls.status AS raw_status
        FROM traffic_light_status AS tls
        WHERE lidar_pc_token IN ({('?,'*len(binary_lpc_tokens))[:-1]});
        """
        df = pd.read_sql_query(query, self.connection, params=binary_lpc_tokens)
        df["status"] = df["raw_status"].map(NUPLAN_TRAFFIC_STATUS_DICT)
        df["lane_id"] = df["lane_id"].astype(str)
        return df.drop(columns=["raw_status"])

    def close_db(self) -> None:
        self.cursor.close()
        self.connection.close()
