import gzip
import json
from pathlib import Path
from typing import Dict, Final, List

import pyarrow as pa
from tqdm import tqdm

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3Index
from asim.common.vehicle_state.ego_vehicle_state import EgoVehicleStateIndex
from asim.dataset.arrow.multiple_table import save_arrow_tables

CARLA_DT: Final[float] = 0.1


def _load_json_gz(path: Path) -> Dict:
    """Helper function to load a gzipped JSON file."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data


class CarlaDataset:
    def __init__(self, output_path: Path, split: str = "carla") -> None:

        self._split: str = split
        self._output_path: Path = output_path
        self._log_path = Path("/home/daniel/carla_workspace/data")

    def convert(self, log_name: str) -> None:

        log_path = self._log_path / f"{log_name}"
        if not log_path.exists():
            raise FileNotFoundError(f"Log path {log_path} does not exist.")

        bounding_box_paths = sorted([bb_path for bb_path in (log_path / "boxes").iterdir()])
        tables: Dict[str, pa.Table] = {}

        tables["metadata_table"] = self._get_metadata_table(_load_json_gz(bounding_box_paths[0])["location"])
        tables["recording_table"] = self._get_recording_table(bounding_box_paths)

        # multi_table = ArrowMultiTableFile(self._output_path / self._split / f"{log_name}.arrow")
        log_file_path = self._output_path / self._split / f"{log_name}.arrow"
        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

        save_arrow_tables(tables, log_file_path)

    def _get_metadata_table(self, location: str) -> pa.Table:
        import asim

        metadata = {
            "dataset": "carla",
            "location": location,
            "vehicle_name": "carla",
            "version": str(asim.__version__),
        }
        metadata_fields = []
        metadata_values = []
        for key, value in metadata.items():
            metadata_fields.append(key)
            metadata_values.append(pa.scalar(value))

        return pa.Table.from_arrays([pa.array([value]) for value in metadata_values], metadata_fields)

    def _get_recording_table(self, bounding_box_paths: List[Path]) -> pa.Table:

        timestamp_log: List[int] = []

        detections_state_log: List[List[List[float]]] = []
        detections_token_log: List[List[str]] = []
        detections_type_log: List[List[int]] = []

        ego_states_log: List[List[float]] = []

        traffic_light_ids_log: List[List[int]] = []
        traffic_light_types_log: List[List[int]] = []
        scenario_tags_log: List[List[str]] = []

        for box_path in tqdm(bounding_box_paths):
            data = _load_json_gz(box_path)
            timestamp_log.append(data["timestamp"])
            detections_state_log.append(data["detections_state"])
            detections_token_log.append(data["detections_token"])
            detections_type_log.append(data["detections_types"])
            ego_states_log.append(data["ego_state"])
            traffic_light_ids_log.append(data["traffic_light_ids"])
            traffic_light_types_log.append(data["traffic_light_types"])
            scenario_tags_log.append(data["scenario_tag"])

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
