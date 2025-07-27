import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool


@dataclass
class DataConverterConfig:

    output_path: Union[str, Path]
    force_log_conversion: bool = False
    force_map_conversion: bool = False
    camera_store_option: Optional[Literal["path", "binary"]] = None
    lidar_store_option: Optional[Literal["path", "binary"]] = None

    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)


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
