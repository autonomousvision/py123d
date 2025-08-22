from __future__ import annotations

from d123.common.utils.enums import SerialIntEnum


class DetectionType(SerialIntEnum):
    """
    Enum for agents in d123.
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


DYNAMIC_DETECTION_TYPES: set[DetectionType] = {
    DetectionType.VEHICLE,
    DetectionType.BICYCLE,
    DetectionType.PEDESTRIAN,
}

STATIC_DETECTION_TYPES: set[DetectionType] = {
    DetectionType.TRAFFIC_CONE,
    DetectionType.BARRIER,
    DetectionType.CZONE_SIGN,
    DetectionType.GENERIC_OBJECT,
}
