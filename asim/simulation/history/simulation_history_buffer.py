from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple, Type

from asim.common.datatypes.recording.abstract_recording import Recording
from asim.common.datatypes.recording.detection_recording import DetectionRecording
from asim.common.datatypes.vehicle_state.ego_state import EgoStateSE2
from asim.dataset.scene.abstract_scene import AbstractScene


class Simulation2DHistoryBuffer:
    """
    This class is used to keep a rolling buffer of a given size. The buffer is a first-in first-out queue. Hence, the
    oldest samples in the buffer are continuously replaced as new samples are appended.
    """

    def __init__(
        self,
        ego_state_buffer: Deque[EgoStateSE2],
        recording_buffer: Deque[Recording],
        sample_interval: Optional[float] = None,
    ):
        """
        Constructs a SimulationHistoryBuffer
        :param ego_state_buffer: Past ego state trajectory including the state.
            at the current time step [t_-N, ..., t_-1, t_0]
        :param observations_buffer: Past observations including the observation.
            at the current time step [t_-N, ..., t_-1, t_0].
        :param sample_interval: [s] the time interval between each sample, if given
        """
        if not ego_state_buffer or not recording_buffer:
            raise ValueError("Ego and observation buffers cannot be empty!")

        if len(ego_state_buffer) != len(recording_buffer):
            raise ValueError(
                "Ego and observations buffer is "
                f"not the same length {len(ego_state_buffer) != len(recording_buffer)}!"
            )

        self._ego_state_buffer = ego_state_buffer
        self._recording_buffer = recording_buffer
        self._sample_interval = sample_interval

    @property
    def ego_state_buffer(self) -> Deque[EgoStateSE2]:
        """
        :return: current ego state buffer
        """
        return self._ego_state_buffer

    @property
    def recording_buffer(self) -> Deque[Recording]:
        """
        :return: current observation buffer
        """
        return self._recording_buffer

    @property
    def size(self) -> int:
        """
        :return: Size of the buffer.
        """
        return len(self.ego_states)

    @property
    def duration(self) -> Optional[float]:
        """
        :return: [s] Duration of the buffer.
        """
        return self.sample_interval * self.size if self.sample_interval else None

    @property
    def current_state(self) -> Tuple[EgoStateSE2, Recording]:
        """
        :return: current state of AV vehicle and its observations
        """
        return self.ego_states[-1], self.recording_buffer[-1]

    @property
    def sample_interval(self) -> Optional[float]:
        """
        :return: the sample interval
        """
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, sample_interval: float) -> None:
        """
        Sets the sample interval of the buffer, raises if the sample interval was not None
        :param sample_interval: The sample interval of the buffer
        """
        assert self._sample_interval is None, "Can't overwrite a pre-existing sample-interval!"
        self._sample_interval = sample_interval

    @property
    def ego_states(self) -> List[EgoStateSE2]:
        """
        :return: the ego state buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        return list(self._ego_state_buffer)

    @property
    def recordings(self) -> List[Recording]:
        """
        :return: the recording buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        return list(self._recording_buffer)

    def append(self, ego_state: EgoStateSE2, recording: Recording) -> None:
        """
        Adds new samples to the buffers
        :param ego_state: an ego state
        :param recording: a recording
        """
        self._ego_state_buffer.append(ego_state)
        self._recording_buffer.append(recording)

    def extend(self, ego_states: List[EgoStateSE2], recordings: List[Recording]) -> None:
        """
        Adds new samples to the buffers
        :param ego_states: an ego states list
        :param recordings: a recordings list
        """
        if len(ego_states) != len(recordings):
            raise ValueError(f"Ego and recordings are not the same length {len(ego_states) != len(recordings)}!")
        self._ego_state_buffer.extend(ego_states)
        self._recording_buffer.extend(recordings)

    def __len__(self) -> int:
        """
        :return: the length of the buffer
        @raise AssertionError if the length of each buffers are not the same
        """
        return len(self._ego_state_buffer)

    @classmethod
    def initialize_from_list(
        cls,
        buffer_size: int,
        ego_states: List[EgoStateSE2],
        recordings: List[Recording],
        sample_interval: Optional[float] = None,
    ) -> Simulation2DHistoryBuffer:
        """
        Create history buffer from lists
        :param buffer_size: size of buffer
        :param ego_states: list of ego states
        :param observations: list of observations
        :param sample_interval: [s] the time interval between each sample, if given
        :return: SimulationHistoryBuffer
        """
        ego_state_buffer: Deque[EgoStateSE2] = deque(ego_states[-buffer_size:], maxlen=buffer_size)
        recording_buffer: Deque[Recording] = deque(recordings[-buffer_size:], maxlen=buffer_size)

        return cls(
            ego_state_buffer=ego_state_buffer, recording_buffer=recording_buffer, sample_interval=sample_interval
        )

    @staticmethod
    def initialize_from_scene(
        buffer_size: int, scene: AbstractScene, recording_type: Type[Recording]
    ) -> Simulation2DHistoryBuffer:
        """
        Initializes ego_state_buffer and recording_buffer from scene
        :param buffer_size: size of the buffer
        :param scene: Simulation scene
        :param recording_type: Recording type used for the simulation
        """
        buffer_duration = buffer_size * scene.database_interval

        if recording_type == DetectionRecording:
            observation_getter = scene.get_past_tracked_objects
        # elif recording_type == Sensors:
        #     observation_getter = scenario.get_past_sensors
        else:
            raise ValueError(f"No matching recording type for {recording_type} for history!")

        past_observation = list(observation_getter(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size))

        past_ego_states = list(
            scene.get_ego_past_trajectory(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size)
        )

        return Simulation2DHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=past_ego_states,
            observations=past_observation,
            sample_interval=scene.database_interval,
        )
