from asim.common.geometry.base import dataclass


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
