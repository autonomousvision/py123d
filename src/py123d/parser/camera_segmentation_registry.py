from __future__ import annotations

from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    DefaultCameraSegmentationLabel,
    register_camera_segmentation_label,
)


@register_camera_segmentation_label
class WODPerceptionCameraSegmentationLabel(CameraSegmentationLabel):
    """WOD-Perception 2D camera panoptic semantic segmentation classes.

    Values are the raw ``CameraSegmentation.Type`` enum from the Waymo Open Dataset spec [1]_
    (29 entries, 28 semantic classes plus ``TYPE_UNDEFINED``).

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/camera_segmentation.proto
    """

    TYPE_UNDEFINED = 0
    TYPE_EGO_VEHICLE = 1
    TYPE_CAR = 2
    TYPE_TRUCK = 3
    TYPE_BUS = 4
    TYPE_OTHER_LARGE_VEHICLE = 5
    TYPE_BICYCLE = 6
    TYPE_MOTORCYCLE = 7
    TYPE_TRAILER = 8
    TYPE_PEDESTRIAN = 9
    TYPE_CYCLIST = 10
    TYPE_MOTORCYCLIST = 11
    TYPE_BIRD = 12
    TYPE_GROUND_ANIMAL = 13
    TYPE_CONSTRUCTION_CONE_POLE = 14
    TYPE_POLE = 15
    TYPE_PEDESTRIAN_OBJECT = 16
    TYPE_SIGN = 17
    TYPE_TRAFFIC_LIGHT = 18
    TYPE_BUILDING = 19
    TYPE_ROAD = 20
    TYPE_LANE_MARKER = 21
    TYPE_ROAD_MARKER = 22
    TYPE_SIDEWALK = 23
    TYPE_VEGETATION = 24
    TYPE_SKY = 25
    TYPE_GROUND = 26
    TYPE_DYNAMIC = 27
    TYPE_STATIC = 28

    def to_default(self) -> DefaultCameraSegmentationLabel:
        """Inherited, see superclass."""
        mapping = {
            WODPerceptionCameraSegmentationLabel.TYPE_UNDEFINED: DefaultCameraSegmentationLabel.IGNORE,
            WODPerceptionCameraSegmentationLabel.TYPE_EGO_VEHICLE: DefaultCameraSegmentationLabel.IGNORE,
            WODPerceptionCameraSegmentationLabel.TYPE_CAR: DefaultCameraSegmentationLabel.VEHICLE,
            WODPerceptionCameraSegmentationLabel.TYPE_TRUCK: DefaultCameraSegmentationLabel.VEHICLE,
            WODPerceptionCameraSegmentationLabel.TYPE_BUS: DefaultCameraSegmentationLabel.VEHICLE,
            WODPerceptionCameraSegmentationLabel.TYPE_OTHER_LARGE_VEHICLE: DefaultCameraSegmentationLabel.VEHICLE,
            WODPerceptionCameraSegmentationLabel.TYPE_BICYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            WODPerceptionCameraSegmentationLabel.TYPE_MOTORCYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            WODPerceptionCameraSegmentationLabel.TYPE_TRAILER: DefaultCameraSegmentationLabel.VEHICLE,
            WODPerceptionCameraSegmentationLabel.TYPE_PEDESTRIAN: DefaultCameraSegmentationLabel.PERSON,
            WODPerceptionCameraSegmentationLabel.TYPE_CYCLIST: DefaultCameraSegmentationLabel.RIDER,
            WODPerceptionCameraSegmentationLabel.TYPE_MOTORCYCLIST: DefaultCameraSegmentationLabel.RIDER,
            WODPerceptionCameraSegmentationLabel.TYPE_BIRD: DefaultCameraSegmentationLabel.OTHER,
            WODPerceptionCameraSegmentationLabel.TYPE_GROUND_ANIMAL: DefaultCameraSegmentationLabel.OTHER,
            WODPerceptionCameraSegmentationLabel.TYPE_CONSTRUCTION_CONE_POLE: DefaultCameraSegmentationLabel.POLE,
            WODPerceptionCameraSegmentationLabel.TYPE_POLE: DefaultCameraSegmentationLabel.POLE,
            WODPerceptionCameraSegmentationLabel.TYPE_PEDESTRIAN_OBJECT: DefaultCameraSegmentationLabel.OTHER,
            WODPerceptionCameraSegmentationLabel.TYPE_SIGN: DefaultCameraSegmentationLabel.TRAFFIC_SIGN,
            WODPerceptionCameraSegmentationLabel.TYPE_TRAFFIC_LIGHT: DefaultCameraSegmentationLabel.TRAFFIC_LIGHT,
            WODPerceptionCameraSegmentationLabel.TYPE_BUILDING: DefaultCameraSegmentationLabel.BUILDING,
            WODPerceptionCameraSegmentationLabel.TYPE_ROAD: DefaultCameraSegmentationLabel.ROAD,
            WODPerceptionCameraSegmentationLabel.TYPE_LANE_MARKER: DefaultCameraSegmentationLabel.ROAD,
            WODPerceptionCameraSegmentationLabel.TYPE_ROAD_MARKER: DefaultCameraSegmentationLabel.ROAD,
            WODPerceptionCameraSegmentationLabel.TYPE_SIDEWALK: DefaultCameraSegmentationLabel.SIDEWALK,
            WODPerceptionCameraSegmentationLabel.TYPE_VEGETATION: DefaultCameraSegmentationLabel.VEGETATION,
            WODPerceptionCameraSegmentationLabel.TYPE_SKY: DefaultCameraSegmentationLabel.SKY,
            WODPerceptionCameraSegmentationLabel.TYPE_GROUND: DefaultCameraSegmentationLabel.TERRAIN,
            WODPerceptionCameraSegmentationLabel.TYPE_DYNAMIC: DefaultCameraSegmentationLabel.OTHER,
            WODPerceptionCameraSegmentationLabel.TYPE_STATIC: DefaultCameraSegmentationLabel.OTHER,
        }
        return mapping[self]
