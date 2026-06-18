"""Static constants and metadata builders for the Griffin parser.

Griffin [1]_ is an aerial-ground cooperative perception dataset collected in
CARLA. This parser converts the **vehicle-side** (ground agent): four cardinal
108.8 deg-FoV RGB cameras (1920x1080) and a single 80-beam top LiDAR at 10 Hz,
together with ego poses and 3D box annotations.

Subsets and splits
------------------
Griffin ships four altitude-stratified subsets. Each subset has its own
official ``train`` / ``val`` partition (there is no ``test`` split); the
partition is defined by scene name in ``split_datas/{subset}.json`` and embedded
in this package under :mod:`py123d.parser.griffin.splits`.

References
----------
.. [1] Wang et al., "Griffin: Aerial-Ground Cooperative Detection and Tracking
   Dataset and Benchmark", arXiv preprint 2503.06983 (2025).
.. [2] Official toolkit: https://github.com/wang-jh18-SVM/Griffin
"""

from __future__ import annotations

from typing import Dict, List

from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.geometry import PoseSE3
from py123d.parser.registry import GriffinBoxDetectionLabel

# ----------------------------------------------------------------------------------------------------------------------
# Subsets and splits
# ----------------------------------------------------------------------------------------------------------------------

# The four released subsets. Each maps to one ``split_datas/{subset}.json`` file.
GRIFFIN_SUBSETS: List[str] = [
    "griffin_50scenes_25m",
    "griffin_50scenes_40m",
    "griffin_50scenes_55m",
    "griffin_100scenes_random",
]

# Griffin only provides train/val partitions (no held-out test split).
GRIFFIN_SPLIT_KINDS: List[str] = ["train", "val"]

# Public split identifiers, e.g. ``griffin_50scenes_25m_train``. The subset
# prefix selects which ``griffin-release`` tree (and embedded split file) to use.
GRIFFIN_SPLITS: List[str] = [f"{subset}_{kind}" for subset in GRIFFIN_SUBSETS for kind in GRIFFIN_SPLIT_KINDS]


def split_to_subset_and_kind(split: str) -> tuple[str, str]:
    """Decompose a Griffin split identifier into ``(subset, kind)``.

    :param split: A split id such as ``"griffin_50scenes_25m_train"``.
    :return: Tuple of ``(subset, kind)``, e.g. ``("griffin_50scenes_25m", "train")``.
    :raises ValueError: If ``split`` is not a recognized Griffin split.
    """
    for subset in GRIFFIN_SUBSETS:
        for kind in GRIFFIN_SPLIT_KINDS:
            if split == f"{subset}_{kind}":
                return subset, kind
    raise ValueError(f"Unrecognized Griffin split '{split}'. Available: {GRIFFIN_SPLITS}")


# ----------------------------------------------------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------------------------------------------------

# Vehicle-side cameras: four wide-FoV cameras in cardinal directions, mapped onto
# the canonical front/back/left/right pinhole ids. The dict keys are the on-disk
# directory and calibration names.
GRIFFIN_VEHICLE_CAMERA_MAPPING: Dict[str, CameraID] = {
    "front": CameraID.PCAM_F0,
    "back": CameraID.PCAM_B0,
    "left": CameraID.PCAM_L0,
    "right": CameraID.PCAM_R0,
}

GRIFFIN_CAMERA_WIDTH: int = 1920
GRIFFIN_CAMERA_HEIGHT: int = 1080

# Griffin LiDAR points are stored pre-merged in the ego frame; we model them as a
# single merged top LiDAR. The on-disk calibration/sensor name is ``lidar_top``.
GRIFFIN_LIDAR_SENSOR_NAME: str = "lidar_top"

# LiDAR sweep period: 80-beam sensor at 10 Hz. Griffin frame ids are 6-digit,
# zero-padded indices spaced 0.1 s apart, so frame ``N`` occurs at
# ``N * GRIFFIN_LIDAR_PERIOD_US`` microseconds (see official converter's
# ``_frame_number_to_nuscenes_timestamp``).
GRIFFIN_LIDAR_PERIOD_US: int = 100_000

# ----------------------------------------------------------------------------------------------------------------------
# Detections
# ----------------------------------------------------------------------------------------------------------------------

# Raw Griffin label type (case-insensitive) -> native Griffin box-detection label.
# Mirrors ``obj_type_mapping`` in the official converter but preserves native
# granularity (truck/bus/motorcycle) instead of collapsing into the 3 benchmark
# classes; the collapse to a unified taxonomy happens in ``to_default``.
# Non-traffic CARLA props (e.g. "Soldier", "Military") are intentionally absent
# and are skipped by the parser rather than mislabeled.
GRIFFIN_BOX_DETECTION_FROM_STR: Dict[str, GriffinBoxDetectionLabel] = {
    "pedestrian": GriffinBoxDetectionLabel.PEDESTRIAN,
    "car": GriffinBoxDetectionLabel.CAR,
    "truck": GriffinBoxDetectionLabel.TRUCK,
    "bus": GriffinBoxDetectionLabel.BUS,
    "motorcycle": GriffinBoxDetectionLabel.MOTORCYCLE,
    "bicycle": GriffinBoxDetectionLabel.BICYCLE,
}

GRIFFIN_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=GriffinBoxDetectionLabel)

# ----------------------------------------------------------------------------------------------------------------------
# Ego vehicle
# ----------------------------------------------------------------------------------------------------------------------

# Griffin's ground agent is a standard CARLA passenger vehicle (Lincoln MKZ
# 2017, CARLA's default ego). The ego frame coincides with the IMU frame, hence
# the identity ``center_to_imu`` / ``rear_axle_to_imu``. Built lazily to keep
# ``EgoStateSE3Metadata`` out of module import time.
_GRIFFIN_EGO_STATE_SE3_METADATA = None


def build_griffin_ego_metadata():
    """Build the (cached) Griffin ego metadata.

    :return: The :class:`~py123d.datatypes.vehicle_state.ego_state_metadata.EgoStateSE3Metadata`
        describing the Griffin ground vehicle.
    """
    global _GRIFFIN_EGO_STATE_SE3_METADATA  # noqa: PLW0603
    if _GRIFFIN_EGO_STATE_SE3_METADATA is None:
        from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata

        _GRIFFIN_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
            vehicle_name="griffin_carla_vehicle",
            width=1.85,
            length=4.79,
            height=1.49,
            wheel_base=2.86,
            center_to_imu_se3=PoseSE3.identity(),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )
    return _GRIFFIN_EGO_STATE_SE3_METADATA
