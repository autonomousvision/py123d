from __future__ import annotations

import abc

from asim.common.vehicle_state.ego_state import EgoVehicleState
from asim.dataset.maps.abstract_map import AbstractMap


class AbstractScene(abc.ABC):
    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        """
        Return the Map API for this scene
        :return: AbstractMap.
        """

    def get_ego_vehicle_state(self, iteration: int) -> EgoVehicleState:
        """
        Get the ego vehicle state at a specific iteration.
        :param iteration: The iteration index.
        :return: EgoVehicleState.
        """
        raise NotImplementedError
