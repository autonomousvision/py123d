import abc
from typing import Iterator

from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.multithreading.worker_utils import WorkerPool


class SceneBuilder(abc.ABC):
    """
    Abstract base class for building scenes from a dataset.
    """

    @abc.abstractmethod
    def get_scenes(self, filter: SceneFilter, worker: WorkerPool) -> Iterator[SceneAPI]:
        """
        Returns an iterator over scenes that match the given filter.
        :param filter: SceneFilter object to filter the scenes.
        :param worker: WorkerPool to parallelize the scene extraction.
        :return: Iterator over AbstractScene objects.
        """
        raise NotImplementedError
