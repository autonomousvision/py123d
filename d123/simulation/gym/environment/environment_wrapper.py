import logging
import sys
import traceback
from threading import Timer
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from d123.simulation.gym.environment.gym_observation.abstract_gym_observation import (
    AbstractGymObservation,
)
from d123.simulation.gym.environment.reward_builder.abstract_reward_builder import AbstractRewardBuilder
from d123.simulation.gym.environment.scenario_sampler.abstract_scenario_sampler import AbstractScenarioSampler
from d123.simulation.gym.environment.simulation_builder.abstract_simulation_builder import (
    AbstractSimulationBuilder,
)
from d123.simulation.gym.environment.simulation_wrapper import SimulationWrapper
from d123.simulation.gym.environment.output_converter.abstract_output_converter import (
    AbstractOutputConverter,
)

logger = logging.getLogger(__name__)


class EnvironmentWrapper(gym.Env):
    """
    Gymnasium environment class interface. Wraps the simulation, trajectory builder, observation builder, and reward builder.
    """

    metadata = {"render_modes": ["rgb_array"]}  # TODO: Figure out the purpose of this metadata.

    def __init__(
        self,
        scenario_sampler: AbstractScenarioSampler,
        simulation_builder: AbstractSimulationBuilder,
        output_converter: AbstractOutputConverter,
        observation_builder: AbstractGymObservation,
        reward_builder: AbstractRewardBuilder,
        terminate_on_failure: bool = False,
    ):
        """
        Initializes the EnvironmentWrapper.
        :param scenario_sampler: Scenario sampler to sample scenarios for the environment.
        :param simulation_builder: Simulation builder to create the simulation from the sampled scenario.
        :param trajectory_builder: Trajectory builder to create trajectories based on actions.
        :param observation_builder: Observation builder to create observations from the simulation state.
        :param reward_builder: Reward builder to create rewards based on the simulation state and actions.
        :param terminate_on_failure: Whether to terminate during an error of the simulation, defaults to False
        """

        self._scenario_sampler = scenario_sampler
        self._simulation_builder = simulation_builder
        self._trajectory_builder = output_converter
        self._observation_builder = observation_builder
        self._reward_builder = reward_builder

        # lazy loaded
        self._simulation_wrapper: Optional[SimulationWrapper] = None

        # timer
        # TODO: Consider removing the timers.
        self._reset_timer = Timer(end_key="reset_total")
        self._step_timer = Timer(end_key="step_total")

        self._terminate_on_error = terminate_on_failure

        # Set for super class
        self.observation_space = observation_builder.get_observation_space()
        self.action_space = output_converter.get_action_space()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Inherited, see superclass."""
        super().reset(seed=seed)
        info: Dict[str, Any] = {}

        try:

            self._reward_builder.reset()
            self._observation_builder.reset()

            self._reset_timer.flush()
            self._reset_timer.start()

            scenario = self._scenario_sampler.sample(seed)
            self._reset_timer.log("reset_1_sample_scenario")

            simulation = self._simulation_builder.build_simulation(scenario)
            self._reset_timer.log("reset_2_build_simulation")

            self._simulation_wrapper = SimulationWrapper(simulation)
            self._reset_timer.log("reset_3_wrap_simulation")

            (
                planner_input,
                planner_initialization,
            ) = self._simulation_wrapper.initialize()
            self._reset_timer.log("reset_4_init_wrapper")

            observation = self._observation_builder.get_gym_observation(planner_input, planner_initialization, info)
            self._reset_timer.log("reset_5_build_observation")
            self._reset_timer.end()

            info["timing"] = self._reset_timer.info()

        except Exception:
            logger.warning(f"{type(self).__name__} failed during .reset() with the following exception:")
            traceback.print_exc()

            if self._terminate_on_error:
                sys.exit(1)
            else:
                observation = create_zero_like_observation(self.observation_space)

        return observation, info

    def step(self, action):
        """Inherited, see superclass."""
        info: Dict[str, Any] = {}

        try:
            assert self._simulation_wrapper is not None

            self._step_timer.flush()
            self._step_timer.start()

            trajectory = self._trajectory_builder.build_trajectory(
                action, self._simulation_wrapper.current_ego_state, info
            )
            self._step_timer.log("step_1_build_trajectory")

            planner_input, is_simulation_running = self._simulation_wrapper.step(trajectory)
            self._step_timer.log("step_2_simulation_step")

            reward, termination, truncation = self._reward_builder.build_reward(self._simulation_wrapper, info)
            termination = termination or not is_simulation_running
            self._step_timer.log("step_3_build_reward")

            observation = self._observation_builder.get_gym_observation(
                planner_input, self._simulation_wrapper.planner_initialization, info
            )

            self._step_timer.log("step_4_build_observation")
            self._step_timer.end()

            info["timing"] = self._step_timer.info()

        except Exception:
            logger.warning(f"{type(self).__name__} failed during .step() with the following exception:")
            traceback.print_exc()

            if self._terminate_on_error:
                sys.exit(1)  # Exit with error code 1
            else:
                observation = create_zero_like_observation(self.observation_space)
                reward = 0.0
                termination = truncation = True

        return observation, reward, termination, truncation, info

    def close(self):
        """Inherited, see superclass."""
        # TODO: Figure out the purpose of this function. :D
        logger.info("EnvironmentWrapper close!")


def create_zero_like_observation(
    observation_space: gym.spaces,
) -> Dict[str, npt.NDArray]:
    """
    TODO: Consider moving elsewhere.
    Creates a zero-like gymnasium observation given the observation space.
    :param observation_space: Gymnasium observation space.
    :raises TypeError: Invalid observation space.
    :return: zero-like gymnasium observation
    """
    if isinstance(observation_space, gym.spaces.Discrete):
        return 0
    elif isinstance(observation_space, gym.spaces.Box):
        return np.zeros(observation_space.shape, dtype=observation_space.dtype)
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return np.zeros(observation_space.shape, dtype=np.int8)
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return np.zeros(observation_space.shape, dtype=np.int64)
    elif isinstance(observation_space, gym.spaces.Dict):
        return {key: create_zero_like_observation(subspace) for key, subspace in observation_space.spaces.items()}
    elif isinstance(observation_space, gym.spaces.Tuple):
        return tuple(create_zero_like_observation(subspace) for subspace in observation_space.spaces)
    else:
        raise TypeError(f"Unsupported space type: {type(observation_space)}")
