# TODO: Remove or implement this placeholder


from dataclasses import dataclass
from typing import List

from d123.conversion.maps.abstract_map import AbstractMap
from d123.simulation.history.simulation_history_buffer import Simulation2DHistoryBuffer
from d123.simulation.time_controller.simulation_iteration import SimulationIteration


@dataclass(frozen=True)
class PlannerInitialization:
    """
    This class represents required data to initialize a planner.
    """

    route_lane_group_ids: List[str]
    map_api: AbstractMap


@dataclass(frozen=True)
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration
    history: Simulation2DHistoryBuffer


class AbstractPlanner:
    def __init__(self):
        self._arg = None

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
