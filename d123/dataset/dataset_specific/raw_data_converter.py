import abc
from typing import List

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool


class RawDataConverter(abc.ABC):

    def __init__(self, force_log_conversion: bool, force_map_conversion: bool) -> None:
        self.force_log_conversion = force_log_conversion
        self.force_map_conversion = force_map_conversion

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
