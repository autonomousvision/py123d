from __future__ import annotations

from dataclasses import asdict, dataclass

from d123.geometry import StateSE2, StateSE3, Vector2D, Vector3D
from d123.geometry.transform.transform_se2 import translate_se2_along_body_frame
from d123.geometry.transform.transform_se3 import translate_se3_along_body_frame


@dataclass
class VehicleParameters:

    vehicle_name: str

    width: float
    length: float
    height: float

    wheel_base: float
    rear_axle_to_center_vertical: float
    rear_axle_to_center_longitudinal: float

    @classmethod
    def from_dict(cls, data_dict: dict) -> VehicleParameters:
        return VehicleParameters(**data_dict)

    def to_dict(self) -> dict:
        return asdict(self)


def get_nuplan_chrysler_pacifica_parameters() -> VehicleParameters:
    # NOTE: use parameters from nuPlan dataset
    return VehicleParameters(
        vehicle_name="nuplan_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=0.45,  # NOTE: missing in nuPlan, TODO: find more accurate value
        rear_axle_to_center_longitudinal=1.461,
    )


def get_carla_lincoln_mkz_2020_parameters() -> VehicleParameters:
    # NOTE: values are extracted from CARLA
    return VehicleParameters(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        rear_axle_to_center_vertical=0.38579,
        rear_axle_to_center_longitudinal=1.64855,
    )


def get_wopd_chrysler_pacifica_parameters() -> VehicleParameters:
    # NOTE: use parameters from nuPlan dataset
    # Find better parameters for WOPD ego vehicle
    return VehicleParameters(
        vehicle_name="wopd_chrysler_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=1.777 / 2,
        rear_axle_to_center_longitudinal=1.461,
    )

def get_kitti360_station_wagon_parameters() -> VehicleParameters:
    #NOTE: Parameters are estimated from the vehicle model.
    #https://www.cvlibs.net/datasets/kitti-360/documentation.php
    return VehicleParameters(
        vehicle_name="kitti360_station_wagon",
        width=1.800,
        length=3.500,
        height=1.400,
        wheel_base=2.710,
        rear_axle_to_center_vertical=0.45,
        rear_axle_to_center_longitudinal=2.71/2 + 0.05,
    )

def get_av2_ford_fusion_hybrid_parameters() -> VehicleParameters:
    # NOTE: Parameters are estimated from the vehicle model.
    # https://en.wikipedia.org/wiki/Ford_Fusion_Hybrid#Second_generation
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


def center_se3_to_rear_axle_se3(center_se3: StateSE3, vehicle_parameters: VehicleParameters) -> StateSE3:
    """
    Converts a center state to a rear axle state.
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


def rear_axle_se3_to_center_se3(rear_axle_se3: StateSE3, vehicle_parameters: VehicleParameters) -> StateSE3:
    """
    Converts a rear axle state to a center state.
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


def center_se2_to_rear_axle_se2(center_se2: StateSE2, vehicle_parameters: VehicleParameters) -> StateSE2:
    """
    Converts a center state to a rear axle state in 2D.
    :param center_se2: The center state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle state in 2D.
    """
    return translate_se2_along_body_frame(center_se2, Vector2D(-vehicle_parameters.rear_axle_to_center_longitudinal, 0))


def rear_axle_se2_to_center_se2(rear_axle_se2: StateSE2, vehicle_parameters: VehicleParameters) -> StateSE2:
    """
    Converts a rear axle state to a center state in 2D.
    :param rear_axle_se2: The rear axle state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center state in 2D.
    """
    return translate_se2_along_body_frame(
        rear_axle_se2, Vector2D(vehicle_parameters.rear_axle_to_center_longitudinal, 0)
    )
