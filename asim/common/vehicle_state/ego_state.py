from dataclasses import dataclass

from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from asim.common.geometry.vector import Vector3D

# TODO: Implement


@dataclass
class CarState:
    def __init__(self):

        self.bounding_box: BoundingBoxSE3 = None
        self.dynamic_state: DynamicCarState = None


@dataclass
class DynamicCarState:
    def __init__(
        self,
    ):
        self.velocity: Vector3D = None
        self.acceleration: Vector3D = None
        self._angular_velocity = None
