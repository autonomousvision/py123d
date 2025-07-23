from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from d123.common.datatypes.recording.detection_recording import DetectionRecording
from d123.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from d123.dataset.scene.abstract_scene import AbstractScene
from d123.simulation.planning.planner_output.abstract_planner_output import AbstractPlannerOutput
from d123.simulation.time_controller.simulation_iteration import SimulationIteration

# from nuplan.common.actor_state.ego_state import EgoState
# from nuplan.common.actor_state.state_representation import StateSE2
# from nuplan.common.maps.abstract_map import AbstractMap
# from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
# from nuplan.planning.simulation.observation.observation_type import Observation
# from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
# from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@dataclass(frozen=True)
class Simulation2DHistorySample:
    """
    Single SimulationHistory sample point.
    """

    iteration: SimulationIteration
    ego_state: EgoStateSE2
    planner_output: AbstractPlannerOutput
    detections: DetectionRecording


class Simulation2DHistory:
    """
    Simulation history including a sequence of simulation states.
    """

    def __init__(self, data: Optional[List[Simulation2DHistorySample]] = None) -> None:
        """
        Construct the history
        :param map_api: abstract map api for accessing the maps
        :param data: A list of simulation data.
        """

        self.data: List[Simulation2DHistorySample] = data if data is not None else list()
        self.scene: Optional[AbstractScene] = None

    def add_sample(self, sample: Simulation2DHistorySample) -> None:
        """
        Add a sample to history
        :param sample: one snapshot of a simulation
        """
        self.data.append(sample)

    def last(self) -> Simulation2DHistorySample:
        """
        :return: last sample from history, or raise if empty
        """
        if not self.data:
            raise RuntimeError("Data is empty!")
        return self.data[-1]

    def reset(self, scene: AbstractScene) -> None:
        """
        Clear the stored data
        """
        self.data.clear()
        self.scene = scene

    def __len__(self) -> int:
        """
        Return the number of history samples as len().
        """
        return len(self.data)

    @property
    def extract_ego_state(self) -> List[EgoStateSE2]:
        """
        Extract ego states in simulation history.
        :return An List of ego_states.
        """
        return [sample.ego_state for sample in self.data]

    @property
    def interval_seconds(self) -> float:
        """
        Return the interval between SimulationHistorySamples.
        :return The interval in seconds.
        """
        if not self.data or len(self.data) < 1:
            raise ValueError("Data is empty!")
        elif len(self.data) < 2:
            raise ValueError("Can't calculate the interval of a single-iteration simulation.")

        return float(self.data[1].iteration.time_s - self.data[0].iteration.time_s)  # float cast is for mypy
