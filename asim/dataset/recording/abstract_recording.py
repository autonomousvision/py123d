import abc
from dataclasses import dataclass
from enum import IntEnum


class RecordingType(IntEnum):
    DETECTION = 0
    # SENSOR = 1 NOTE: not used yet, but reserved for future use


@dataclass
class Recording(abc.ABC):
    """
    Abstract observation container.
    """

    # @classmethod
    # @abc.abstractmethod
    # def observation_type(cls) -> ObservationType:
    #     """
    #     Returns detection type of the observation.
    #     """
    #     raise NotImplementedError
