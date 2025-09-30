import abc
from typing import Iterator

from d123.common.multithreading.worker_utils import WorkerPool
from d123.datatypes.scene.abstract_scene import AbstractScene
from d123.datatypes.scene.scene_filter import SceneFilter

# TODO: Expand lazy implementation for scene builder.


class SceneBuilder(abc.ABC):
    """
    Abstract base class for building scenes from a dataset.
    """

    @abc.abstractmethod
    def get_scenes(self, filter: SceneFilter, worker: WorkerPool) -> Iterator[AbstractScene]:
        """
        Returns an iterator over scenes that match the given filter.
        :param filter: SceneFilter object to filter the scenes.
        :param worker: WorkerPool to parallelize the scene extraction.
        :return: Iterator over AbstractScene objects.
        """
        raise NotImplementedError
