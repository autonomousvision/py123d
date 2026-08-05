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
    def get_scenes(self, filter: SceneFilter, executor: Executor, lazy: bool = False) -> Sequence[SceneAPI]:
        """Returns the scenes that match the given filter.

        :param filter: SceneFilter object to filter the scenes.
        :param executor: Executor to parallelize the scene extraction.
        :param lazy: Whether to build each scene when it is indexed rather than up front. Enumerating a
            large split this way costs a few arrays per log instead of an object per scene.
        :return: The matching scenes, as a list unless *lazy* is set.
        """
