import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

from d123.common.multithreading.worker_utils import WorkerPool


@dataclass
class DataConverterConfig:

    output_path: Union[str, Path]
    force_log_conversion: bool = False
    force_map_conversion: bool = False

    # Ego
    include_ego: bool = False

    # Box Detections
    include_box_detections: bool = False

    # Traffic Lights
    include_traffic_lights: bool = False

    # Cameras
    include_cameras: bool = False
    camera_store_option: Literal["path", "binary", "mp4"] = "path"

    # LiDARs
    include_lidars: bool = False
    lidar_store_option: Literal["path", "binary"] = "path"

    # Scenario tag / Route
    # NOTE: These are only supported for nuPlan. Consider removing or expanding support.
    include_scenario_tags: bool = False
    include_route: bool = False

    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        assert self.camera_store_option != "mp4", "MP4 format is not yet supported."
        assert self.camera_store_option in [
            "path",
            "binary",
        ], f"Invalid camera store option, got {self.camera_store_option}."

        assert self.lidar_store_option in [
            "path",
            "binary",
        ], f"Invalid LiDAR store option, got {self.lidar_store_option}."


class RawDataConverter(abc.ABC):

    def __init__(self, data_converter_config: DataConverterConfig) -> None:
        self.data_converter_config = data_converter_config

    @abc.abstractmethod
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""

    @abc.abstractmethod
    def convert_maps(self, worker: WorkerPool) -> None:
        """
        Convert maps in raw data format to the uniform 123D format.
        :param worker: The worker pool to use for parallel processing.
        """

    @abc.abstractmethod
    def convert_logs(self, worker: WorkerPool) -> None:
        """
        Convert logs in raw data format to the uniform 123D format.
        :param worker: The worker pool to use for parallel processing.
        """
