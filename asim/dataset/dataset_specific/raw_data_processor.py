import abc
from typing import List

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool


class RawDataProcessor(abc.ABC):

    def __init__(self, force_data_conversion: bool) -> None:
        self.force_data_conversion = force_data_conversion

    @abc.abstractmethod
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""

    @abc.abstractmethod
    def convert(self, worker: WorkerPool) -> None:
        pass
