from __future__ import annotations

from py123d.common.utils.enums import SerialIntEnum

BOX_DETECTION_TYPE_REGISTRY = {}


def register_box_detection_type(enum_class):
    BOX_DETECTION_TYPE_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


class BoxDetectionType(SerialIntEnum):
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


# @register_box_detection_type
# class NuPlanBoxDetectionType(SerialIntEnum):

#     VEHICLE = 0
#     BICYCLE = 1
#     PEDESTRIAN = 2
#     TRAFFIC_CONE = 3
#     BARRIER = 4
#     CZONE_SIGN = 5
#     GENERIC_OBJECT = 6

#     def to_default_type() -> BoxDetectionType:
#         mapping = {
#             NuPlanBoxDetectionType.VEHICLE: BoxDetectionType.VEHICLE,
#             NuPlanBoxDetectionType.BICYCLE: BoxDetectionType.BICYCLE,
#             NuPlanBoxDetectionType.PEDESTRIAN: BoxDetectionType.PEDESTRIAN,
#             NuPlanBoxDetectionType.TRAFFIC_CONE: BoxDetectionType.GENERIC_OBJECT,
#             NuPlanBoxDetectionType.BARRIER: BoxDetectionType.GENERIC_OBJECT,
#             NuPlanBoxDetectionType.CZONE_SIGN: BoxDetectionType.GENERIC_OBJECT,
#             NuPlanBoxDetectionType.GENERIC_OBJECT: BoxDetectionType.GENERIC_OBJECT,
#         }
#         return mapping[self]
