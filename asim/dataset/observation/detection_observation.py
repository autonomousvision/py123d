import abc
from dataclasses import dataclass


@dataclass
class Observation(abc.ABC):
    """
    Abstract observation container.
    """

    @classmethod
    def detection_type(cls) -> str:
        """
        Returns detection type of the observation.
        """
        return cls.__name__


@dataclass
class SensorObservation(Observation):
    pass


@dataclass
class DetectionObservation(Observation):
    """
    Output of the perception system, i.e. tracks.
    """

    # tracked_objects: TrackedObjects
