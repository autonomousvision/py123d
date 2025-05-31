from __future__ import annotations

from asim.common.utils.enums import SerialIntEnum


class DetectionType(SerialIntEnum):
    """
    Enum for agents in asim.
    """

    VEHICLE = 0  # Includes all four or more wheeled vehicles, as well as trailers.
    BICYCLE = 1  # Includes bicycles, motorcycles and tricycles.
    PEDESTRIAN = 2  # Pedestrians, incl. strollers and wheelchairs.

    TRAFFIC_CONE = 3  # Cones that are temporarily placed to control the flow of traffic.
    BARRIER = 4  # Solid barriers that can be either temporary or permanent.
    CZONE_SIGN = 5  # Temporary signs that indicate construction zones.
    GENERIC_OBJECT = 6  # Animals, debris, pushable/pullable objects, permanent poles.
