from __future__ import annotations

import abc

from py123d.common.utils.enums import SerialIntEnum

BOX_DETECTION_LABEL_REGISTRY = {}


def register_box_detection_label(enum_class):
    BOX_DETECTION_LABEL_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


class BoxDetectionLabel(SerialIntEnum):

    @abc.abstractmethod
    def to_default(self) -> DefaultBoxDetectionLabel:
        raise NotImplementedError("Subclasses must implement this method.")


@register_box_detection_label
class DefaultBoxDetectionLabel(BoxDetectionLabel):
    """
    Enum for agents in py123d.
    """

    VEHICLE = 0
    BICYCLE = 1
    PEDESTRIAN = 2

    TRAFFIC_CONE = 3
    BARRIER = 4
    CZONE_SIGN = 5
    GENERIC_OBJECT = 6

    EGO = 7
    SIGN = 8  # TODO: Remove or extent

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        return self


@register_box_detection_label
class AV2SensorBoxDetectionLabel(BoxDetectionLabel):
    """Sensor dataset annotation categories."""

    ANIMAL = 1
    ARTICULATED_BUS = 2
    BICYCLE = 3
    BICYCLIST = 4
    BOLLARD = 5
    BOX_TRUCK = 6
    BUS = 7
    CONSTRUCTION_BARREL = 8
    CONSTRUCTION_CONE = 9
    DOG = 10
    LARGE_VEHICLE = 11
    MESSAGE_BOARD_TRAILER = 12
    MOBILE_PEDESTRIAN_CROSSING_SIGN = 13
    MOTORCYCLE = 14
    MOTORCYCLIST = 15
    OFFICIAL_SIGNALER = 16
    PEDESTRIAN = 17
    RAILED_VEHICLE = 18
    REGULAR_VEHICLE = 19
    SCHOOL_BUS = 20
    SIGN = 21
    STOP_SIGN = 22
    STROLLER = 23
    TRAFFIC_LIGHT_TRAILER = 24
    TRUCK = 25
    TRUCK_CAB = 26
    VEHICULAR_TRAILER = 27
    WHEELCHAIR = 28
    WHEELED_DEVICE = 29
    WHEELED_RIDER = 30

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            AV2SensorBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            AV2SensorBoxDetectionLabel.ARTICULATED_BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            AV2SensorBoxDetectionLabel.BICYCLIST: DefaultBoxDetectionLabel.PEDESTRIAN,
            AV2SensorBoxDetectionLabel.BOLLARD: DefaultBoxDetectionLabel.BARRIER,
            AV2SensorBoxDetectionLabel.BOX_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_BARREL: DefaultBoxDetectionLabel.BARRIER,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            AV2SensorBoxDetectionLabel.DOG: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            AV2SensorBoxDetectionLabel.LARGE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MESSAGE_BOARD_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MOBILE_PEDESTRIAN_CROSSING_SIGN: DefaultBoxDetectionLabel.CZONE_SIGN,
            AV2SensorBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            AV2SensorBoxDetectionLabel.MOTORCYCLIST: DefaultBoxDetectionLabel.BICYCLE,
            AV2SensorBoxDetectionLabel.OFFICIAL_SIGNALER: DefaultBoxDetectionLabel.PEDESTRIAN,
            AV2SensorBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PEDESTRIAN,
            AV2SensorBoxDetectionLabel.RAILED_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.REGULAR_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SCHOOL_BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SIGN: DefaultBoxDetectionLabel.SIGN,
            AV2SensorBoxDetectionLabel.STOP_SIGN: DefaultBoxDetectionLabel.SIGN,
            AV2SensorBoxDetectionLabel.STROLLER: DefaultBoxDetectionLabel.PEDESTRIAN,
            AV2SensorBoxDetectionLabel.TRAFFIC_LIGHT_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK_CAB: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.VEHICULAR_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.WHEELCHAIR: DefaultBoxDetectionLabel.PEDESTRIAN,
            AV2SensorBoxDetectionLabel.WHEELED_DEVICE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            AV2SensorBoxDetectionLabel.WHEELED_RIDER: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]


@register_box_detection_label
class KITTI360BoxDetectionLabel(BoxDetectionLabel):

    BICYCLE = 0
    BOX = 1
    BUS = 2
    CAR = 3
    CARAVAN = 4
    LAMP = 5
    MOTORCYCLE = 6
    PERSON = 7
    POLE = 8
    RIDER = 9
    SMALLPOLE = 10
    STOP = 11
    TRAFFIC_LIGHT = 12
    TRAFFIC_SIGN = 13
    TRAILER = 14
    TRAIN = 15
    TRASH_BIN = 16
    TRUCK = 17
    VENDING_MACHINE = 18

    def to_default(self) -> DefaultBoxDetectionLabel:
        mapping = {
            KITTI360BoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            KITTI360BoxDetectionLabel.BOX: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CARAVAN: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.LAMP: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            KITTI360BoxDetectionLabel.PERSON: DefaultBoxDetectionLabel.PEDESTRIAN,
            KITTI360BoxDetectionLabel.POLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.RIDER: DefaultBoxDetectionLabel.BICYCLE,
            KITTI360BoxDetectionLabel.SMALLPOLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.STOP: DefaultBoxDetectionLabel.SIGN,
            KITTI360BoxDetectionLabel.TRAFFIC_LIGHT: DefaultBoxDetectionLabel.SIGN,
            KITTI360BoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.SIGN,
            KITTI360BoxDetectionLabel.TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.TRASH_BIN: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.VENDING_MACHINE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class NuPlanBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for nuPlan bounding box detections.

    Descriptions in `.db` files:
    - vehicle: Includes all four or more wheeled vehicles, as well as trailers.
    - bicycle: Includes bicycles, motorcycles and tricycles.
    - pedestrian: All types of pedestrians, incl. strollers and wheelchairs.
    - traffic_cone: Cones that are temporarily placed to control the flow of traffic.
    - barrier: Solid barriers that can be either temporary or permanent.
    - czone_sign: Temporary signs that indicate construction zones.
    - generic_object: Animals, debris, pushable/pullable objects, permanent poles.
    """

    VEHICLE = 0
    BICYCLE = 1
    PEDESTRIAN = 2
    TRAFFIC_CONE = 3
    BARRIER = 4
    CZONE_SIGN = 5
    GENERIC_OBJECT = 6

    def to_default(self) -> DefaultBoxDetectionLabel:
        mapping = {
            NuPlanBoxDetectionLabel.VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            NuPlanBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuPlanBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuPlanBoxDetectionLabel.TRAFFIC_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NuPlanBoxDetectionLabel.BARRIER: DefaultBoxDetectionLabel.BARRIER,
            NuPlanBoxDetectionLabel.CZONE_SIGN: DefaultBoxDetectionLabel.CZONE_SIGN,
            NuPlanBoxDetectionLabel.GENERIC_OBJECT: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class NuScenesBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for nuScenes bounding box detections.
    [1] https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md#labels
    """

    VEHICLE_CAR = 0
    VEHICLE_TRUCK = 1
    VEHICLE_BUS_BENDY = 2
    VEHICLE_BUS_RIGID = 3
    VEHICLE_CONSTRUCTION = 4
    VEHICLE_EMERGENCY_AMBULANCE = 5
    VEHICLE_EMERGENCY_POLICE = 6
    VEHICLE_TRAILER = 7
    VEHICLE_BICYCLE = 8
    VEHICLE_MOTORCYCLE = 9
    HUMAN_PEDESTRIAN_ADULT = 10
    HUMAN_PEDESTRIAN_CHILD = 11
    HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER = 12
    HUMAN_PEDESTRIAN_PERSONAL_MOBILITY = 13
    HUMAN_PEDESTRIAN_POLICE_OFFICER = 14
    HUMAN_PEDESTRIAN_STROLLER = 15
    HUMAN_PEDESTRIAN_WHEELCHAIR = 16
    MOVABLE_OBJECT_TRAFFICCONE = 17
    MOVABLE_OBJECT_BARRIER = 18
    MOVABLE_OBJECT_PUSHABLE_PULLABLE = 19
    MOVABLE_OBJECT_DEBRIS = 20
    STATIC_OBJECT_BICYCLE_RACK = 21
    ANIMAL = 22

    def to_default(self):
        mapping = {
            NuScenesBoxDetectionLabel.VEHICLE_CAR: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BUS_BENDY: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BUS_RIGID: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_CONSTRUCTION: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_AMBULANCE: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CHILD: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_PERSONAL_MOBILITY: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_POLICE_OFFICER: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_WHEELCHAIR: DefaultBoxDetectionLabel.PEDESTRIAN,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_TRAFFICCONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_BARRIER: DefaultBoxDetectionLabel.BARRIER,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_PUSHABLE_PULLABLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_DEBRIS: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.STATIC_OBJECT_BICYCLE_RACK: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class PandasetBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for Pandaset bounding box detections.
    [1] https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_cuboids.pdf
    """

    ANIMALS_BIRD = 0
    ANIMALS_OTHER = 1
    BICYCLE = 2
    BUS = 3
    CAR = 4
    CONES = 5
    CONSTRUCTION_SIGNS = 6
    EMERGENCY_VEHICLE = 7
    MEDIUM_SIZED_TRUCK = 8
    MOTORCYCLE = 9
    MOTORIZED_SCOOTER = 10
    OTHER_VEHICLE_CONSTRUCTION_VEHICLE = 11
    OTHER_VEHICLE_PEDICAB = 12
    OTHER_VEHICLE_UNCOMMON = 13
    PEDESTRIAN = 14
    PEDESTRIAN_WITH_OBJECT = 15
    PERSONAL_MOBILITY_DEVICE = 16
    PICKUP_TRUCK = 17
    PYLONS = 18
    ROAD_BARRIERS = 19
    ROLLING_CONTAINERS = 20
    SEMI_TRUCK = 21
    SIGNS = 22
    TEMPORARY_CONSTRUCTION_BARRIERS = 23
    TOWED_OBJECT = 24
    TRAIN = 25
    TRAM_SUBWAY = 26

    def to_default(self) -> DefaultBoxDetectionLabel:
        mapping = {
            PandasetBoxDetectionLabel.ANIMALS_BIRD: DefaultBoxDetectionLabel.GENERIC_OBJECT,  # TODO: Adjust default types
            PandasetBoxDetectionLabel.ANIMALS_OTHER: DefaultBoxDetectionLabel.GENERIC_OBJECT,  # TODO: Adjust default types
            PandasetBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CONES: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.CONSTRUCTION_SIGNS: DefaultBoxDetectionLabel.CZONE_SIGN,
            PandasetBoxDetectionLabel.EMERGENCY_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MEDIUM_SIZED_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.MOTORIZED_SCOOTER: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_CONSTRUCTION_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_PEDICAB: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_UNCOMMON: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PEDESTRIAN,
            PandasetBoxDetectionLabel.PEDESTRIAN_WITH_OBJECT: DefaultBoxDetectionLabel.PEDESTRIAN,
            PandasetBoxDetectionLabel.PERSONAL_MOBILITY_DEVICE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.PICKUP_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PYLONS: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.ROAD_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.ROLLING_CONTAINERS: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            PandasetBoxDetectionLabel.SEMI_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.SIGNS: DefaultBoxDetectionLabel.SIGN,
            PandasetBoxDetectionLabel.TEMPORARY_CONSTRUCTION_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.TOWED_OBJECT: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.GENERIC_OBJECT,  # TODO: Adjust default types
            PandasetBoxDetectionLabel.TRAM_SUBWAY: DefaultBoxDetectionLabel.GENERIC_OBJECT,  # TODO: Adjust default types
        }
        return mapping[self]


@register_box_detection_label
class WOPDBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for Waymo Open Dataset bounding box detections.
    [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md
    [2] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63-L69
    """

    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4

    def to_default(self) -> DefaultBoxDetectionLabel:
        mapping = {
            WOPDBoxDetectionLabel.TYPE_UNKNOWN: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            WOPDBoxDetectionLabel.TYPE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            WOPDBoxDetectionLabel.TYPE_PEDESTRIAN: DefaultBoxDetectionLabel.PEDESTRIAN,
            WOPDBoxDetectionLabel.TYPE_SIGN: DefaultBoxDetectionLabel.SIGN,
            WOPDBoxDetectionLabel.TYPE_CYCLIST: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]
