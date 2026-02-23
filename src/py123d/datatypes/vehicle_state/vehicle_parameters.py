from __future__ import annotations

from dataclasses import asdict, dataclass

from py123d.geometry import PoseSE2, PoseSE3, Vector2D, Vector3D
from py123d.geometry.transform import translate_se2_along_body_frame, translate_se3_along_body_frame


@dataclass
class VehicleParameters:
    """Parameters that describe the physical dimensions of a vehicle."""

    vehicle_name: str
    """Name of the vehicle model."""

    width: float
    """Width of the vehicle."""

    length: float
    """Length of the vehicle."""

    height: float
    """Height of the vehicle."""

    wheel_base: float
    """Wheel base of the vehicle (longitudinal distance between front and rear axles)."""

    rear_axle_to_center_vertical: float
    """Distance from the rear axle to the center of the vehicle (vertical)."""

    rear_axle_to_center_longitudinal: float
    """Distance from the rear axle to the center of the vehicle (longitudinal)."""

    @classmethod
    def from_dict(cls, data_dict: dict) -> VehicleParameters:
        """Creates a VehicleParameters instance from a dictionary.

        :param data_dict: Dictionary containing vehicle parameters.
        :return: VehicleParameters instance.
        """
        return VehicleParameters(**data_dict)

    @property
    def half_width(self) -> float:
        """Half the width of the vehicle."""
        return self.width / 2.0

    @property
    def half_length(self) -> float:
        """Half the length of the vehicle."""
        return self.length / 2.0

    @property
    def half_height(self) -> float:
        """Half the height of the vehicle."""
        return self.height / 2.0

    def to_dict(self) -> dict:
        """Converts the :class:`VehicleParameters` instance to a dictionary.

        :return: Dictionary representation of the vehicle parameters.
        """
        return asdict(self)


def get_nuplan_chrysler_pacifica_parameters() -> VehicleParameters:
    """Helper function to get nuPlan Chrysler Pacifica vehicle parameters."""
    # NOTE: These parameters are mostly available in nuPlan, except for the rear_axle_to_center_vertical.
    # The value is estimated based the LiDAR point cloud.
    # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
    return VehicleParameters(
        vehicle_name="nuplan_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=0.45,
        rear_axle_to_center_longitudinal=1.461,
    )


def get_nuscenes_renault_zoe_parameters() -> VehicleParameters:
    """Helper function to get nuScenes Renault Zoe vehicle parameters."""
    # NOTE: The parameters in nuScenes are estimates, and partially taken from the Renault Zoe model [1].
    # [1] https://en.wikipedia.org/wiki/Renault_Zoe
    return VehicleParameters(
        vehicle_name="nuscenes_renault_zoe",
        width=1.730,
        length=4.084,
        height=1.562,
        wheel_base=2.588,
        rear_axle_to_center_vertical=1.562 / 2,
        rear_axle_to_center_longitudinal=1.385,
    )


def get_carla_lincoln_mkz_2020_parameters() -> VehicleParameters:
    """Helper function to get CARLA Lincoln MKZ 2020 vehicle parameters."""
    # NOTE: These parameters are taken from the CARLA simulator vehicle model. The rear axles to center transform
    # parameters are calculated based on parameters from the CARLA simulator.
    return VehicleParameters(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        rear_axle_to_center_vertical=0.38579,
        rear_axle_to_center_longitudinal=1.64855,
    )


def get_wod_perception_chrysler_pacifica_parameters() -> VehicleParameters:
    """Helper function to get Waymo Open Dataset Perception Chrysler Pacifica vehicle parameters."""
    # NOTE: These parameters are estimates based on the vehicle model used in the WOD Perception dataset.
    # The vehicle should be the same (or a similar) vehicle model to nuPlan and PandaSet [1].
    # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
    return VehicleParameters(
        vehicle_name="wod_perception_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=1.777 / 2,
        rear_axle_to_center_longitudinal=1.461,
    )


def get_wod_motion_chrysler_pacifica_parameters() -> VehicleParameters:
    return VehicleParameters(
        vehicle_name="wod-motion_chrysler_pacifica",
        width=2.3320000171661377,
        length=5.285999774932861,
        height=2.3299999237060547,
        wheel_base=3.089,
        rear_axle_to_center_vertical=2.3299999237060547 / 2,
        rear_axle_to_center_longitudinal=1.461,
    )


def get_kitti360_vw_passat_parameters() -> VehicleParameters:
    """Helper function to get KITTI-360 VW Passat vehicle parameters."""
    # NOTE: The parameters in KITTI-360 are estimates based on the vehicle model used in the dataset
    # Uses a 2006 VW Passat Variant B6 [1]. Vertical distance is estimated based on the LiDAR.
    # KITTI-360 is currently the only dataset where the IMU has a lateral offset to the rear axle [2]
    # We do account for such offsets, but the overall estimations are not perfect.
    # [1] https://en.wikipedia.org/wiki/Volkswagen_Passat_(B6)
    # [2] https://www.cvlibs.net/datasets/kitti-360/documentation.php
    return VehicleParameters(
        vehicle_name="kitti360_vw_passat",
        width=1.820,
        length=4.775,
        height=1.516,
        wheel_base=2.709,
        rear_axle_to_center_vertical=1.516 / 2 - 0.9,
        rear_axle_to_center_longitudinal=1.3369,
    )


def get_av2_ford_fusion_hybrid_parameters() -> VehicleParameters:
    """Helper function to get Argoverse 2 Ford Fusion Hybrid vehicle parameters."""
    # NOTE: Parameters are estimated from the vehicle model [1] and LiDAR point cloud.
    # [1] https://en.wikipedia.org/wiki/Ford_Fusion_Hybrid#Second_generation
    # https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/tests/unit/map/test_map_api.py#L375
    return VehicleParameters(
        vehicle_name="av2_ford_fusion_hybrid",
        width=1.852 + 0.275,  # 0.275 is the estimated width of the side mirrors
        length=4.869,
        height=1.476,
        wheel_base=2.850,
        rear_axle_to_center_vertical=0.438,
        rear_axle_to_center_longitudinal=1.339,
    )


def get_pandaset_chrysler_pacifica_parameters() -> VehicleParameters:
    """Helper function to get PandaSet Chrysler Pacifica vehicle parameters."""
    # NOTE: Some parameters are available in PandaSet [1], others are estimated based on the vehicle model [2].
    # [1] https://arxiv.org/pdf/2112.12610 (Figure 3 (a))
    # [2] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
    return VehicleParameters(
        vehicle_name="pandaset_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=0.45,
        rear_axle_to_center_longitudinal=1.461,
    )


def center_se3_to_rear_axle_se3(center_se3: PoseSE3, vehicle_parameters: VehicleParameters) -> PoseSE3:
    """Converts a center state to a rear axle state.

    :param center_se3: The center state.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle state.
    """
    return translate_se3_along_body_frame(
        center_se3,
        Vector3D(
            -vehicle_parameters.rear_axle_to_center_longitudinal,
            0,
            -vehicle_parameters.rear_axle_to_center_vertical,
        ),
    )


def rear_axle_se3_to_center_se3(rear_axle_se3: PoseSE3, vehicle_parameters: VehicleParameters) -> PoseSE3:
    """Converts a rear axle state to a center state.

    :param rear_axle_se3: The rear axle state.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center state.
    """
    return translate_se3_along_body_frame(
        rear_axle_se3,
        Vector3D(
            vehicle_parameters.rear_axle_to_center_longitudinal,
            0,
            vehicle_parameters.rear_axle_to_center_vertical,
        ),
    )


def center_se2_to_rear_axle_se2(center_se2: PoseSE2, vehicle_parameters: VehicleParameters) -> PoseSE2:
    """Converts a center state to a rear axle state in 2D.

    :param center_se2: The center state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle state in 2D.
    """
    return translate_se2_along_body_frame(center_se2, Vector2D(-vehicle_parameters.rear_axle_to_center_longitudinal, 0))


def rear_axle_se2_to_center_se2(rear_axle_se2: PoseSE2, vehicle_parameters: VehicleParameters) -> PoseSE2:
    """Converts a rear axle state to a center state in 2D.

    :param rear_axle_se2: The rear axle state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center state in 2D.
    """
    return translate_se2_along_body_frame(
        rear_axle_se2, Vector2D(vehicle_parameters.rear_axle_to_center_longitudinal, 0)
    )
