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


@register_camera_segmentation_label
class Kitti360CameraSegmentationLabel(CameraSegmentationLabel):
    """KITTI-360 2D camera semantic segmentation classes (45 classes, ids ``0..44``).

    Values are the raw ``id`` field from the KITTI-360 label definition [1]_ (a Cityscapes-derived
    taxonomy), stored verbatim in the single-channel ``image_0{0,1}/semantic/*.png`` label maps. The
    ``license plate`` class (id ``-1``) is omitted as it never appears in the released label maps.

    The mapping to :class:`DefaultCameraSegmentationLabel` follows the KITTI-360 ``category`` field:
    ``flat`` → ROAD/SIDEWALK, ``construction`` → BUILDING (the unified taxonomy has no dedicated
    wall/fence/bridge class), ``object`` → POLE/TRAFFIC_LIGHT/TRAFFIC_SIGN (street furniture without a
    counterpart → OTHER), ``nature`` → VEGETATION/TERRAIN, ``sky`` → SKY, ``human`` → PERSON/RIDER,
    ``vehicle`` → VEHICLE/TWO_WHEELER, and ``void`` → IGNORE (with the few semantically meaningful void
    classes bucketed by closest fit).

    References
    ----------
    .. [1] https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
    """

    UNLABELED = 0
    EGO_VEHICLE = 1
    RECTIFICATION_BORDER = 2
    OUT_OF_ROI = 3
    STATIC = 4
    DYNAMIC = 5
    GROUND = 6
    ROAD = 7
    SIDEWALK = 8
    PARKING = 9
    RAIL_TRACK = 10
    BUILDING = 11
    WALL = 12
    FENCE = 13
    GUARD_RAIL = 14
    BRIDGE = 15
    TUNNEL = 16
    POLE = 17
    POLEGROUP = 18
    TRAFFIC_LIGHT = 19
    TRAFFIC_SIGN = 20
    VEGETATION = 21
    TERRAIN = 22
    SKY = 23
    PERSON = 24
    RIDER = 25
    CAR = 26
    TRUCK = 27
    BUS = 28
    CARAVAN = 29
    TRAILER = 30
    TRAIN = 31
    MOTORCYCLE = 32
    BICYCLE = 33
    GARAGE = 34
    GATE = 35
    STOP = 36
    SMALLPOLE = 37
    LAMP = 38
    TRASH_BIN = 39
    VENDING_MACHINE = 40
    BOX = 41
    UNKNOWN_CONSTRUCTION = 42
    UNKNOWN_VEHICLE = 43
    UNKNOWN_OBJECT = 44

    def to_default(self) -> DefaultCameraSegmentationLabel:
        """Inherited, see superclass."""
        mapping = {
            # void — unlabeled / ego / out-of-frame / generic static.
            Kitti360CameraSegmentationLabel.UNLABELED: DefaultCameraSegmentationLabel.IGNORE,
            Kitti360CameraSegmentationLabel.EGO_VEHICLE: DefaultCameraSegmentationLabel.IGNORE,
            Kitti360CameraSegmentationLabel.RECTIFICATION_BORDER: DefaultCameraSegmentationLabel.IGNORE,
            Kitti360CameraSegmentationLabel.OUT_OF_ROI: DefaultCameraSegmentationLabel.IGNORE,
            Kitti360CameraSegmentationLabel.STATIC: DefaultCameraSegmentationLabel.IGNORE,
            Kitti360CameraSegmentationLabel.DYNAMIC: DefaultCameraSegmentationLabel.OTHER,
            Kitti360CameraSegmentationLabel.GROUND: DefaultCameraSegmentationLabel.TERRAIN,
            # flat.
            Kitti360CameraSegmentationLabel.ROAD: DefaultCameraSegmentationLabel.ROAD,
            Kitti360CameraSegmentationLabel.SIDEWALK: DefaultCameraSegmentationLabel.SIDEWALK,
            Kitti360CameraSegmentationLabel.PARKING: DefaultCameraSegmentationLabel.ROAD,
            Kitti360CameraSegmentationLabel.RAIL_TRACK: DefaultCameraSegmentationLabel.OTHER,
            # construction — no dedicated wall/fence/bridge class in the unified taxonomy.
            Kitti360CameraSegmentationLabel.BUILDING: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.WALL: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.FENCE: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.GUARD_RAIL: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.BRIDGE: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.TUNNEL: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.GARAGE: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.GATE: DefaultCameraSegmentationLabel.BUILDING,
            Kitti360CameraSegmentationLabel.STOP: DefaultCameraSegmentationLabel.BUILDING,
            # object — poles / signals / street furniture.
            Kitti360CameraSegmentationLabel.POLE: DefaultCameraSegmentationLabel.POLE,
            Kitti360CameraSegmentationLabel.POLEGROUP: DefaultCameraSegmentationLabel.POLE,
            Kitti360CameraSegmentationLabel.SMALLPOLE: DefaultCameraSegmentationLabel.POLE,
            Kitti360CameraSegmentationLabel.LAMP: DefaultCameraSegmentationLabel.POLE,
            Kitti360CameraSegmentationLabel.TRAFFIC_LIGHT: DefaultCameraSegmentationLabel.TRAFFIC_LIGHT,
            Kitti360CameraSegmentationLabel.TRAFFIC_SIGN: DefaultCameraSegmentationLabel.TRAFFIC_SIGN,
            Kitti360CameraSegmentationLabel.TRASH_BIN: DefaultCameraSegmentationLabel.OTHER,
            Kitti360CameraSegmentationLabel.VENDING_MACHINE: DefaultCameraSegmentationLabel.OTHER,
            Kitti360CameraSegmentationLabel.BOX: DefaultCameraSegmentationLabel.OTHER,
            # nature.
            Kitti360CameraSegmentationLabel.VEGETATION: DefaultCameraSegmentationLabel.VEGETATION,
            Kitti360CameraSegmentationLabel.TERRAIN: DefaultCameraSegmentationLabel.TERRAIN,
            # sky.
            Kitti360CameraSegmentationLabel.SKY: DefaultCameraSegmentationLabel.SKY,
            # human.
            Kitti360CameraSegmentationLabel.PERSON: DefaultCameraSegmentationLabel.PERSON,
            Kitti360CameraSegmentationLabel.RIDER: DefaultCameraSegmentationLabel.RIDER,
            # vehicle.
            Kitti360CameraSegmentationLabel.CAR: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.TRUCK: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.BUS: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.CARAVAN: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.TRAILER: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.TRAIN: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.MOTORCYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            Kitti360CameraSegmentationLabel.BICYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            # void — unknown but semantically meaningful.
            Kitti360CameraSegmentationLabel.UNKNOWN_CONSTRUCTION: DefaultCameraSegmentationLabel.OTHER,
            Kitti360CameraSegmentationLabel.UNKNOWN_VEHICLE: DefaultCameraSegmentationLabel.VEHICLE,
            Kitti360CameraSegmentationLabel.UNKNOWN_OBJECT: DefaultCameraSegmentationLabel.OTHER,
        }
        return mapping[self]
