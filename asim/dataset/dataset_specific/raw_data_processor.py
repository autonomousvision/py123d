import abc
from typing import List

from nuplan.planning.utils.multithreading.worker_utils import WorkerPool


class RawDataProcessor(abc.ABC):
    pass

    @abc.abstractmethod
    def get_available_splits(self) -> List[str]:
        """Returns a list of available raw data types."""

    @abc.abstractmethod
    def convert(self, worker: WorkerPool) -> None:
        pass
