from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import shapely

from asim.common.geometry.base import Point2D, Point3D, Point3DIndex, StateSE2
from asim.common.geometry.transform.tranform_2d import translate_along_yaw
from asim.common.geometry.utils import normalize_angle
from asim.dataset.dataset_specific.carla.opendrive.elements.objects import Object
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border

# TODO: make naming consistent with group_collections.py


@dataclass
class OpenDriveObjectHelper:

    object_id: int
    outline_3d: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        assert self.outline_3d.ndim == 2
        assert self.outline_3d.shape[1] == len(Point3DIndex)

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        return shapely.geometry.Polygon(self.outline_3d[:, Point3DIndex.XY])


def get_object_helper(object: Object, reference_border: Border) -> OpenDriveObjectHelper:

    object_helper: Optional[OpenDriveObjectHelper] = None

    # 1. Extract object position in frenet frame of the reference line

    object_se2: StateSE2 = StateSE2.from_array(reference_border.interpolate_se2(s=object.s, t=object.t))
    object_3d: Point3D = Point3D.from_array(reference_border.interpolate_3d(s=object.s, t=object.t))

    # Adjust yaw angle from object data
    object_se2.yaw = normalize_angle(object_se2.yaw + object.hdg)

    if len(object.outline) == 0:
        outline_3d = np.zeros((4, len(Point3DIndex)), dtype=np.float64)

        # Fill XY
        outline_3d[0, Point3DIndex.XY] = translate_along_yaw(
            object_se2, Point2D(object.length / 2.0, object.width / 2.0)
        ).point_2d.array
        outline_3d[1, Point3DIndex.XY] = translate_along_yaw(
            object_se2, Point2D(object.length / 2.0, -object.width / 2.0)
        ).point_2d.array
        outline_3d[2, Point3DIndex.XY] = translate_along_yaw(
            object_se2, Point2D(-object.length / 2.0, -object.width / 2.0)
        ).point_2d.array
        outline_3d[3, Point3DIndex.XY] = translate_along_yaw(
            object_se2, Point2D(-object.length / 2.0, object.width / 2.0)
        ).point_2d.array

        # Fill Z
        outline_3d[..., Point3DIndex.Z] = object_3d.z + object.z_offset
        object_helper = OpenDriveObjectHelper(object_id=object.id, outline_3d=outline_3d)

    else:
        assert len(object.outline) > 3, f"Object outline must have at least 3 corners, got {len(object.outline)}"
        outline_3d = np.zeros((len(object.outline), len(Point3DIndex)), dtype=np.float64)
        for corner_idx, corner_local in enumerate(object.outline):
            outline_3d[corner_idx, Point3DIndex.XY] = translate_along_yaw(
                object_se2, Point2D(corner_local.u, corner_local.v)
            ).point_2d.array
            outline_3d[corner_idx, Point3DIndex.Z] = object_3d.z + corner_local.z
        object_helper = OpenDriveObjectHelper(object_id=object.id, outline_3d=outline_3d)

    return object_helper
