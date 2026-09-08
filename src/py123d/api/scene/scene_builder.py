import abc
from typing import Sequence

from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import Executor


class SceneBuilder(abc.ABC):
    """Base class for all scene builders. The scene builder is responsible for building scene given a \
        :class:`~py123d.api.scene.scene_filter.SceneFilter`.
    """

    @abc.abstractmethod
    def get_scenes(self, filter: SceneFilter, executor: Executor) -> Sequence[SceneAPI]:
        """Returns the scenes that match the given filter.

        A sequence rather than a list, so an implementation may build its scenes
        on access instead of up front.

        :param filter: SceneFilter object to filter the scenes.
        :param executor: Executor to parallelize the scene extraction.
        :return: The matching scenes.
        """
