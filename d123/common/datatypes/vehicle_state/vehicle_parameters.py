from d123.common.geometry.base import StateSE2, StateSE3, dataclass
from d123.common.geometry.transform.se3 import translate_se3_along_x, translate_se3_along_z
from d123.common.geometry.transform.tranform_2d import translate_along_yaw
from d123.common.geometry.vector import Vector2D

# TODO: Add more vehicle parameters, potentially extend the parameters.


@dataclass
class VehicleParameters:

    vehicle_name: str

    width: float
    length: float
    height: float

    wheel_base: float
    rear_axle_to_center_vertical: float
    rear_axle_to_center_longitudinal: float


def get_nuplan_pacifica_parameters() -> VehicleParameters:
    # NOTE: use parameters from nuPlan dataset
    return VehicleParameters(
        vehicle_name="nuplan_pacifica",
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


def get_wopd_pacifica_parameters() -> VehicleParameters:
    # NOTE: use parameters from nuPlan dataset
    # Find better parameters for WOPD ego vehicle
    return VehicleParameters(
        vehicle_name="wopd_pacifica",
        width=2.297,
        length=5.176,
        height=1.777,
        wheel_base=3.089,
        rear_axle_to_center_vertical=0.45,
        rear_axle_to_center_longitudinal=1.461,
    )


def center_se3_to_rear_axle_se3(center_se3: StateSE3, vehicle_parameters: VehicleParameters) -> StateSE3:
    """
    Converts a center state to a rear axle state.
    :param center_se3: The center state.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle state.
    """
    return translate_se3_along_z(
        translate_se3_along_x(
            center_se3,
            -vehicle_parameters.rear_axle_to_center_longitudinal,
        ),
        -vehicle_parameters.rear_axle_to_center_vertical,
    )


def rear_axle_se3_to_center_se3(rear_axle_se3: StateSE3, vehicle_parameters: VehicleParameters) -> StateSE3:
    """
    Converts a rear axle state to a center state.
    :param rear_axle_se3: The rear axle state.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center state.
    """
    return translate_se3_along_x(
        translate_se3_along_z(
            rear_axle_se3,
            vehicle_parameters.rear_axle_to_center_vertical,
        ),
        vehicle_parameters.rear_axle_to_center_longitudinal,
    )


def center_se2_to_rear_axle_se2(center_se2: StateSE2, vehicle_parameters: VehicleParameters) -> StateSE2:
    """
    Converts a center state to a rear axle state in 2D.
    :param center_se2: The center state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The rear axle state in 2D.
    """
    return translate_along_yaw(center_se2, Vector2D(-vehicle_parameters.rear_axle_to_center_longitudinal, 0))


def rear_axle_se2_to_center_se2(rear_axle_se2: StateSE2, vehicle_parameters: VehicleParameters) -> StateSE2:
    """
    Converts a rear axle state to a center state in 2D.
    :param rear_axle_se2: The rear axle state in 2D.
    :param vehicle_parameters: The vehicle parameters.
    :return: The center state in 2D.
    """
    return translate_along_yaw(rear_axle_se2, Vector2D(vehicle_parameters.rear_axle_to_center_longitudinal, 0))
