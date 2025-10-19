from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import shapely

from py123d.conversion.utils.map_utils.opendrive.parser.objects import Object
from py123d.conversion.utils.map_utils.opendrive.parser.reference import ReferenceLine
from py123d.geometry import Point3D, Point3DIndex, StateSE2, Vector2D
from py123d.geometry.geometry_index import StateSE2Index
from py123d.geometry.polyline import Polyline3D
from py123d.geometry.transform.transform_se2 import translate_se2_along_body_frame
from py123d.geometry.utils.rotation_utils import normalize_angle


@dataclass
class OpenDriveObjectHelper:

    object_id: int
    outline_3d: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        assert self.outline_3d.ndim == 2
        assert self.outline_3d.shape[1] == len(Point3DIndex)

    @property
    def outline_polyline_3d(self) -> Polyline3D:
        return Polyline3D.from_array(self.outline_3d)

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        return shapely.geometry.Polygon(self.outline_3d[:, Point3DIndex.XY])


def get_object_helper(object: Object, reference_line: ReferenceLine) -> OpenDriveObjectHelper:

    object_helper: Optional[OpenDriveObjectHelper] = None

    # 1. Extract object position in frenet frame of the reference line

    object_se2: StateSE2 = StateSE2.from_array(reference_line.interpolate_se2(s=object.s, t=object.t))
    object_3d: Point3D = Point3D.from_array(reference_line.interpolate_3d(s=object.s, t=object.t))

    # Adjust yaw angle from object data
    # TODO: Consider adding setters to StateSE2 to make this cleaner
    object_se2._array[StateSE2Index.YAW] = normalize_angle(object_se2.yaw + object.hdg)

    if len(object.outline) == 0:
        outline_3d = np.zeros((4, len(Point3DIndex)), dtype=np.float64)

        # Fill XY
        outline_3d[0, Point3DIndex.XY] = translate_se2_along_body_frame(
            object_se2, Vector2D(object.length / 2.0, object.width / 2.0)
        ).point_2d.array
        outline_3d[1, Point3DIndex.XY] = translate_se2_along_body_frame(
            object_se2, Vector2D(object.length / 2.0, -object.width / 2.0)
        ).point_2d.array
        outline_3d[2, Point3DIndex.XY] = translate_se2_along_body_frame(
            object_se2, Vector2D(-object.length / 2.0, -object.width / 2.0)
        ).point_2d.array
        outline_3d[3, Point3DIndex.XY] = translate_se2_along_body_frame(
            object_se2, Vector2D(-object.length / 2.0, object.width / 2.0)
        ).point_2d.array

        # Fill Z
        outline_3d[..., Point3DIndex.Z] = object_3d.z + object.z_offset
        object_helper = OpenDriveObjectHelper(object_id=object.id, outline_3d=outline_3d)

    else:
        assert len(object.outline) > 3, f"Object outline must have at least 3 corners, got {len(object.outline)}"
        outline_3d = np.zeros((len(object.outline), len(Point3DIndex)), dtype=np.float64)
        for corner_idx, corner_local in enumerate(object.outline):
            outline_3d[corner_idx, Point3DIndex.XY] = translate_se2_along_body_frame(
                object_se2, Vector2D(corner_local.u, corner_local.v)
            ).point_2d.array
            outline_3d[corner_idx, Point3DIndex.Z] = object_3d.z + corner_local.z
        object_helper = OpenDriveObjectHelper(object_id=object.id, outline_3d=outline_3d)

    return object_helper
