from __future__ import annotations

from importlib import abc

from py123d.common.utils.enums import SerialIntEnum


class AbstractBoxDetectionType(SerialIntEnum):

    @abc.abstractmethod
    def to_default_type(self):
        raise NotImplementedError("Subclasses must implement this method.")


class BoxDetectionType(AbstractBoxDetectionType):
    """
    Enum for agents in py123d.
    """

    # TODO:
    # - Add detection types compatible with other datasets
    # - Add finer detection types (e.g. bicycle, motorcycle) and add generic types (e.g. two-wheeled vehicle) for general use.

    # NOTE: Current types strongly aligned with nuPlan.

    VEHICLE = 0  # Includes all four or more wheeled vehicles, as well as trailers.
    BICYCLE = 1  # Includes bicycles, motorcycles and tricycles.
    PEDESTRIAN = 2  # Pedestrians, incl. strollers and wheelchairs.

    TRAFFIC_CONE = 3  # Cones that are temporarily placed to control the flow of traffic.
    BARRIER = 4  # Solid barriers that can be either temporary or permanent.
    CZONE_SIGN = 5  # Temporary signs that indicate construction zones.
    GENERIC_OBJECT = 6  # Animals, debris, pushable/pullable objects, permanent poles.

    EGO = 7
    SIGN = 8  # TODO: Remove or extent

    def to_default_type(self):
        """Inherited, see superclass."""
        return self


DYNAMIC_DETECTION_TYPES: set[BoxDetectionType] = {
    BoxDetectionType.VEHICLE,
    BoxDetectionType.BICYCLE,
    BoxDetectionType.PEDESTRIAN,
}

STATIC_DETECTION_TYPES: set[BoxDetectionType] = {
    BoxDetectionType.TRAFFIC_CONE,
    BoxDetectionType.BARRIER,
    BoxDetectionType.CZONE_SIGN,
    BoxDetectionType.GENERIC_OBJECT,
}
