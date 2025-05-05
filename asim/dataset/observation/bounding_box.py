from dataclasses import dataclass

import shapely

from asim.common.geometry.base import Point2D, Point3D, StateSE2
from asim.common.geometry.tranform_2d import translate_along_yaw


@dataclass
class BoundingBox3D:

    center: Point3D
    yaw: float
    length: float
    width: float
    height: float

    @property
    def state_se2(self):
        return StateSE2(self.center.x, self.center.y, self.yaw)

    @property
    def shapely_polygon(self) -> shapely.geometry.Polygon:

        return shapely.geometry.Polygon(
            [
                translate_along_yaw(self.state_se2, Point2D(self.length / 2.0, self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.state_se2, Point2D(self.length / 2.0, -self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.state_se2, Point2D(-self.length / 2.0, -self.width / 2.0)).point_2d.array,
                translate_along_yaw(self.state_se2, Point2D(-self.length / 2.0, self.width / 2.0)).point_2d.array,
            ]
        )
