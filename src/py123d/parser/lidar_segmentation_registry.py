from __future__ import annotations

from py123d.datatypes.sensors.lidar_segmentation_label import (
    DefaultLidarSegmentationLabel,
    LidarSegmentationLabel,
    register_lidar_segmentation_label,
)


@register_lidar_segmentation_label
class WODPerceptionLidarSegmentationLabel(LidarSegmentationLabel):
    """WOD-Perception 3D LiDAR semantic segmentation classes.

    Values are the raw ``Segmentation.Type`` enum from the Waymo Open Dataset spec [1]_ (23 classes).

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto
    """

    TYPE_UNDEFINED = 0
    TYPE_CAR = 1
    TYPE_TRUCK = 2
    TYPE_BUS = 3
    TYPE_OTHER_VEHICLE = 4
    TYPE_MOTORCYCLIST = 5
    TYPE_BICYCLIST = 6
    TYPE_PEDESTRIAN = 7
    TYPE_SIGN = 8
    TYPE_TRAFFIC_LIGHT = 9
    TYPE_POLE = 10
    TYPE_CONSTRUCTION_CONE = 11
    TYPE_BICYCLE = 12
    TYPE_MOTORCYCLE = 13
    TYPE_BUILDING = 14
    TYPE_VEGETATION = 15
    TYPE_TREE_TRUNK = 16
    TYPE_CURB = 17
    TYPE_ROAD = 18
    TYPE_LANE_MARKER = 19
    TYPE_OTHER_GROUND = 20
    TYPE_WALKABLE = 21
    TYPE_SIDEWALK = 22

    def to_default(self) -> DefaultLidarSegmentationLabel:
        """Inherited, see superclass."""
        mapping = {
            WODPerceptionLidarSegmentationLabel.TYPE_UNDEFINED: DefaultLidarSegmentationLabel.IGNORE,
            WODPerceptionLidarSegmentationLabel.TYPE_CAR: DefaultLidarSegmentationLabel.VEHICLE,
            WODPerceptionLidarSegmentationLabel.TYPE_TRUCK: DefaultLidarSegmentationLabel.VEHICLE,
            WODPerceptionLidarSegmentationLabel.TYPE_BUS: DefaultLidarSegmentationLabel.VEHICLE,
            WODPerceptionLidarSegmentationLabel.TYPE_OTHER_VEHICLE: DefaultLidarSegmentationLabel.VEHICLE,
            WODPerceptionLidarSegmentationLabel.TYPE_MOTORCYCLIST: DefaultLidarSegmentationLabel.RIDER,
            WODPerceptionLidarSegmentationLabel.TYPE_BICYCLIST: DefaultLidarSegmentationLabel.RIDER,
            WODPerceptionLidarSegmentationLabel.TYPE_PEDESTRIAN: DefaultLidarSegmentationLabel.PERSON,
            WODPerceptionLidarSegmentationLabel.TYPE_SIGN: DefaultLidarSegmentationLabel.MANMADE,
            WODPerceptionLidarSegmentationLabel.TYPE_TRAFFIC_LIGHT: DefaultLidarSegmentationLabel.MANMADE,
            WODPerceptionLidarSegmentationLabel.TYPE_POLE: DefaultLidarSegmentationLabel.MANMADE,
            WODPerceptionLidarSegmentationLabel.TYPE_CONSTRUCTION_CONE: DefaultLidarSegmentationLabel.TRAFFIC_CONE,
            WODPerceptionLidarSegmentationLabel.TYPE_BICYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            WODPerceptionLidarSegmentationLabel.TYPE_MOTORCYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            WODPerceptionLidarSegmentationLabel.TYPE_BUILDING: DefaultLidarSegmentationLabel.MANMADE,
            WODPerceptionLidarSegmentationLabel.TYPE_VEGETATION: DefaultLidarSegmentationLabel.VEGETATION,
            WODPerceptionLidarSegmentationLabel.TYPE_TREE_TRUNK: DefaultLidarSegmentationLabel.VEGETATION,
            WODPerceptionLidarSegmentationLabel.TYPE_CURB: DefaultLidarSegmentationLabel.SIDEWALK,
            WODPerceptionLidarSegmentationLabel.TYPE_ROAD: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            WODPerceptionLidarSegmentationLabel.TYPE_LANE_MARKER: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            WODPerceptionLidarSegmentationLabel.TYPE_OTHER_GROUND: DefaultLidarSegmentationLabel.TERRAIN,
            WODPerceptionLidarSegmentationLabel.TYPE_WALKABLE: DefaultLidarSegmentationLabel.TERRAIN,
            WODPerceptionLidarSegmentationLabel.TYPE_SIDEWALK: DefaultLidarSegmentationLabel.SIDEWALK,
        }
        return mapping[self]


@register_lidar_segmentation_label
class PandasetLidarSegmentationLabel(LidarSegmentationLabel):
    """PandaSet 3D LiDAR semantic segmentation classes.

    Values are the raw class ids from PandaSet's per-log ``annotations/semseg/classes.json`` (42
    classes, ids ``1..42``; the file is identical across all annotated logs). PandaSet has no
    unlabeled/sentinel class — every point in an annotated frame carries a label, so there is no
    ``0`` entry. Semantic segmentation is provided for a subset of logs only.

    The mapping to :class:`DefaultLidarSegmentationLabel` is a best-effort coarsening; several
    PandaSet classes have no exact counterpart in the unified taxonomy (e.g. ``Pylons`` →
    ``TRAFFIC_CONE``, ``Driveway`` → ``DRIVABLE_SURFACE``, ``Other Static Object`` → ``OTHER``) and
    are bucketed by closest semantic fit rather than a documented PandaSet↔nuScenes correspondence.

    References
    ----------
    .. [1] https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_semantic_segmentation.pdf
    """

    SMOKE = 1
    EXHAUST = 2
    SPRAY_OR_RAIN = 3
    REFLECTION = 4
    VEGETATION = 5
    GROUND = 6
    ROAD = 7
    LANE_LINE_MARKING = 8
    STOP_LINE_MARKING = 9
    OTHER_ROAD_MARKING = 10
    SIDEWALK = 11
    DRIVEWAY = 12
    CAR = 13
    PICKUP_TRUCK = 14
    MEDIUM_SIZED_TRUCK = 15
    SEMI_TRUCK = 16
    TOWED_OBJECT = 17
    MOTORCYCLE = 18
    OTHER_VEHICLE_CONSTRUCTION_VEHICLE = 19
    OTHER_VEHICLE_UNCOMMON = 20
    OTHER_VEHICLE_PEDICAB = 21
    EMERGENCY_VEHICLE = 22
    BUS = 23
    PERSONAL_MOBILITY_DEVICE = 24
    MOTORIZED_SCOOTER = 25
    BICYCLE = 26
    TRAIN = 27
    TROLLEY = 28
    TRAM_SUBWAY = 29
    PEDESTRIAN = 30
    PEDESTRIAN_WITH_OBJECT = 31
    ANIMALS_BIRD = 32
    ANIMALS_OTHER = 33
    PYLONS = 34
    ROAD_BARRIERS = 35
    SIGNS = 36
    CONES = 37
    CONSTRUCTION_SIGNS = 38
    TEMPORARY_CONSTRUCTION_BARRIERS = 39
    ROLLING_CONTAINERS = 40
    BUILDING = 41
    OTHER_STATIC_OBJECT = 42

    def to_default(self) -> DefaultLidarSegmentationLabel:
        """Inherited, see superclass."""
        mapping = {
            # Noise / atmospheric artifacts — ignored.
            PandasetLidarSegmentationLabel.SMOKE: DefaultLidarSegmentationLabel.IGNORE,
            PandasetLidarSegmentationLabel.EXHAUST: DefaultLidarSegmentationLabel.IGNORE,
            PandasetLidarSegmentationLabel.SPRAY_OR_RAIN: DefaultLidarSegmentationLabel.IGNORE,
            PandasetLidarSegmentationLabel.REFLECTION: DefaultLidarSegmentationLabel.IGNORE,
            # Ground / flat surfaces.
            PandasetLidarSegmentationLabel.VEGETATION: DefaultLidarSegmentationLabel.VEGETATION,
            PandasetLidarSegmentationLabel.GROUND: DefaultLidarSegmentationLabel.TERRAIN,
            PandasetLidarSegmentationLabel.ROAD: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            PandasetLidarSegmentationLabel.LANE_LINE_MARKING: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            PandasetLidarSegmentationLabel.STOP_LINE_MARKING: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            PandasetLidarSegmentationLabel.OTHER_ROAD_MARKING: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            PandasetLidarSegmentationLabel.SIDEWALK: DefaultLidarSegmentationLabel.SIDEWALK,
            PandasetLidarSegmentationLabel.DRIVEWAY: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            # Four-wheeled / large vehicles.
            PandasetLidarSegmentationLabel.CAR: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.PICKUP_TRUCK: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.MEDIUM_SIZED_TRUCK: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.SEMI_TRUCK: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.TOWED_OBJECT: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.OTHER_VEHICLE_CONSTRUCTION_VEHICLE: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.OTHER_VEHICLE_UNCOMMON: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.OTHER_VEHICLE_PEDICAB: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.EMERGENCY_VEHICLE: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.BUS: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.TRAIN: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.TROLLEY: DefaultLidarSegmentationLabel.VEHICLE,
            PandasetLidarSegmentationLabel.TRAM_SUBWAY: DefaultLidarSegmentationLabel.VEHICLE,
            # Small / two-wheeled personal vehicles.
            PandasetLidarSegmentationLabel.MOTORCYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            PandasetLidarSegmentationLabel.PERSONAL_MOBILITY_DEVICE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            PandasetLidarSegmentationLabel.MOTORIZED_SCOOTER: DefaultLidarSegmentationLabel.TWO_WHEELER,
            PandasetLidarSegmentationLabel.BICYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            # People.
            PandasetLidarSegmentationLabel.PEDESTRIAN: DefaultLidarSegmentationLabel.PERSON,
            PandasetLidarSegmentationLabel.PEDESTRIAN_WITH_OBJECT: DefaultLidarSegmentationLabel.PERSON,
            # Animals — no dedicated default class.
            PandasetLidarSegmentationLabel.ANIMALS_BIRD: DefaultLidarSegmentationLabel.OTHER,
            PandasetLidarSegmentationLabel.ANIMALS_OTHER: DefaultLidarSegmentationLabel.OTHER,
            # Road furniture / static structures.
            PandasetLidarSegmentationLabel.PYLONS: DefaultLidarSegmentationLabel.TRAFFIC_CONE,
            PandasetLidarSegmentationLabel.CONES: DefaultLidarSegmentationLabel.TRAFFIC_CONE,
            PandasetLidarSegmentationLabel.ROAD_BARRIERS: DefaultLidarSegmentationLabel.BARRIER,
            PandasetLidarSegmentationLabel.TEMPORARY_CONSTRUCTION_BARRIERS: DefaultLidarSegmentationLabel.BARRIER,
            PandasetLidarSegmentationLabel.SIGNS: DefaultLidarSegmentationLabel.MANMADE,
            PandasetLidarSegmentationLabel.CONSTRUCTION_SIGNS: DefaultLidarSegmentationLabel.MANMADE,
            PandasetLidarSegmentationLabel.BUILDING: DefaultLidarSegmentationLabel.MANMADE,
            PandasetLidarSegmentationLabel.ROLLING_CONTAINERS: DefaultLidarSegmentationLabel.OTHER,
            PandasetLidarSegmentationLabel.OTHER_STATIC_OBJECT: DefaultLidarSegmentationLabel.OTHER,
        }
        return mapping[self]


@register_lidar_segmentation_label
class NuScenesLidarSegmentationLabel(LidarSegmentationLabel):
    """nuScenes per-point semantic segmentation classes (32 fine-grained classes, ids ``0..31``).

    Values are the ``index`` field of the nuScenes-lidarseg ``category.json`` and match the canonical
    order in the devkit's ``nuscenes/utils/color_map.py`` (``noise`` … ``vehicle.ego``). The labels are
    derived from nuScenes-**panoptic** (``panoptic_label = category_idx * 1000 + instance_id``), so the
    per-point semantic id is ``panoptic // 1000``; the same 32-class taxonomy backs both lidarseg and
    panoptic. The mapping to :class:`DefaultLidarSegmentationLabel` follows the official lidarseg
    "challenge" coarsening, with the fine ``human.pedestrian.*`` subclasses collapsed to ``PERSON`` and
    the ego-vehicle self-returns / noise treated as ``IGNORE``.

    References
    ----------
    .. [1] https://www.nuscenes.org/nuscenes#lidarseg
    .. [2] https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
    """

    NOISE = 0
    ANIMAL = 1
    PEDESTRIAN_ADULT = 2
    PEDESTRIAN_CHILD = 3
    PEDESTRIAN_CONSTRUCTION_WORKER = 4
    PEDESTRIAN_PERSONAL_MOBILITY = 5
    PEDESTRIAN_POLICE_OFFICER = 6
    PEDESTRIAN_STROLLER = 7
    PEDESTRIAN_WHEELCHAIR = 8
    BARRIER = 9
    DEBRIS = 10
    PUSHABLE_PULLABLE = 11
    TRAFFICCONE = 12
    BICYCLE_RACK = 13
    BICYCLE = 14
    BUS_BENDY = 15
    BUS_RIGID = 16
    CAR = 17
    CONSTRUCTION_VEHICLE = 18
    EMERGENCY_AMBULANCE = 19
    EMERGENCY_POLICE = 20
    MOTORCYCLE = 21
    TRAILER = 22
    TRUCK = 23
    DRIVEABLE_SURFACE = 24
    OTHER_FLAT = 25
    SIDEWALK = 26
    TERRAIN = 27
    MANMADE = 28
    STATIC_OTHER = 29
    VEGETATION = 30
    EGO = 31

    def to_default(self) -> DefaultLidarSegmentationLabel:
        """Inherited, see superclass."""
        mapping = {
            # Noise / ego self-returns — ignored.
            NuScenesLidarSegmentationLabel.NOISE: DefaultLidarSegmentationLabel.IGNORE,
            NuScenesLidarSegmentationLabel.EGO: DefaultLidarSegmentationLabel.IGNORE,
            # People (all pedestrian subclasses, incl. personal mobility / stroller / wheelchair).
            NuScenesLidarSegmentationLabel.PEDESTRIAN_ADULT: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_CHILD: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_CONSTRUCTION_WORKER: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_PERSONAL_MOBILITY: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_POLICE_OFFICER: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_STROLLER: DefaultLidarSegmentationLabel.PERSON,
            NuScenesLidarSegmentationLabel.PEDESTRIAN_WHEELCHAIR: DefaultLidarSegmentationLabel.PERSON,
            # Two-wheeled vehicles.
            NuScenesLidarSegmentationLabel.BICYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            NuScenesLidarSegmentationLabel.MOTORCYCLE: DefaultLidarSegmentationLabel.TWO_WHEELER,
            # Four-wheeled / large vehicles.
            NuScenesLidarSegmentationLabel.BUS_BENDY: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.BUS_RIGID: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.CAR: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.CONSTRUCTION_VEHICLE: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.EMERGENCY_AMBULANCE: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.EMERGENCY_POLICE: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.TRAILER: DefaultLidarSegmentationLabel.VEHICLE,
            NuScenesLidarSegmentationLabel.TRUCK: DefaultLidarSegmentationLabel.VEHICLE,
            # Road furniture / barriers.
            NuScenesLidarSegmentationLabel.BARRIER: DefaultLidarSegmentationLabel.BARRIER,
            NuScenesLidarSegmentationLabel.TRAFFICCONE: DefaultLidarSegmentationLabel.TRAFFIC_CONE,
            NuScenesLidarSegmentationLabel.BICYCLE_RACK: DefaultLidarSegmentationLabel.MANMADE,
            NuScenesLidarSegmentationLabel.MANMADE: DefaultLidarSegmentationLabel.MANMADE,
            # Ground / flat surfaces.
            NuScenesLidarSegmentationLabel.DRIVEABLE_SURFACE: DefaultLidarSegmentationLabel.DRIVABLE_SURFACE,
            NuScenesLidarSegmentationLabel.SIDEWALK: DefaultLidarSegmentationLabel.SIDEWALK,
            NuScenesLidarSegmentationLabel.TERRAIN: DefaultLidarSegmentationLabel.TERRAIN,
            NuScenesLidarSegmentationLabel.VEGETATION: DefaultLidarSegmentationLabel.VEGETATION,
            # Miscellaneous — no exact default counterpart.
            NuScenesLidarSegmentationLabel.ANIMAL: DefaultLidarSegmentationLabel.OTHER,
            NuScenesLidarSegmentationLabel.DEBRIS: DefaultLidarSegmentationLabel.OTHER,
            NuScenesLidarSegmentationLabel.PUSHABLE_PULLABLE: DefaultLidarSegmentationLabel.OTHER,
            NuScenesLidarSegmentationLabel.OTHER_FLAT: DefaultLidarSegmentationLabel.OTHER,
            NuScenesLidarSegmentationLabel.STATIC_OTHER: DefaultLidarSegmentationLabel.OTHER,
        }
        return mapping[self]
