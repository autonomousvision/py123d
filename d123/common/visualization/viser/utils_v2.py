import numpy as np
import numpy.typing as npt

from d123.common.visualization.color.default import BOX_DETECTION_CONFIG, EGO_VEHICLE_CONFIG
from d123.common.visualization.viser.utils import BRIGHTNESS_FACTOR
from d123.dataset.scene.abstract_scene import AbstractScene

# from d123.common.datatypes.sensor.camera_parameters import get_nuplan_camera_parameters
from d123.geometry import BoundingBoxSE3, Corners3DIndex, Point3D, Point3DIndex, Vector3D
from d123.geometry.transform.transform_se3 import translate_se3_along_body_frame

# TODO: Refactor this file.
# TODO: Add general utilities for 3D primitives and mesh support.


def _get_bounding_box_corners(bounding_box: BoundingBoxSE3) -> npt.NDArray[np.float64]:
    """
    Get the vertices of a bounding box in 3D space.
    """
    corner_extent_factors = {
        Corners3DIndex.FRONT_LEFT_BOTTOM: Vector3D(+0.5, -0.5, -0.5),
        Corners3DIndex.FRONT_RIGHT_BOTTOM: Vector3D(+0.5, +0.5, -0.5),
        Corners3DIndex.BACK_RIGHT_BOTTOM: Vector3D(-0.5, +0.5, -0.5),
        Corners3DIndex.BACK_LEFT_BOTTOM: Vector3D(-0.5, -0.5, -0.5),
        Corners3DIndex.FRONT_LEFT_TOP: Vector3D(+0.5, -0.5, +0.5),
        Corners3DIndex.FRONT_RIGHT_TOP: Vector3D(+0.5, +0.5, +0.5),
        Corners3DIndex.BACK_RIGHT_TOP: Vector3D(-0.5, +0.5, +0.5),
        Corners3DIndex.BACK_LEFT_TOP: Vector3D(-0.5, -0.5, +0.5),
    }
    corners = np.zeros((len(Corners3DIndex), len(Point3DIndex)), dtype=np.float64)
    bounding_box_extent = np.array([bounding_box.length, bounding_box.width, bounding_box.height], dtype=np.float64)
    for idx, vec in corner_extent_factors.items():
        vector_3d = Vector3D.from_array(bounding_box_extent * vec.array)
        corners[idx] = translate_se3_along_body_frame(bounding_box.center, vector_3d).point_3d.array
    return corners


def _get_bounding_box_lines(bounding_box: BoundingBoxSE3) -> npt.NDArray[np.float64]:
    """
    Get the edges of a bounding box in 3D space as a Polyline3D.
    """
    corners = _get_bounding_box_corners(bounding_box)
    index_pairs = [
        (Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.FRONT_RIGHT_BOTTOM),
        (Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.BACK_RIGHT_BOTTOM),
        (Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.BACK_LEFT_BOTTOM),
        (Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.FRONT_LEFT_BOTTOM),
        (Corners3DIndex.FRONT_LEFT_TOP, Corners3DIndex.FRONT_RIGHT_TOP),
        (Corners3DIndex.FRONT_RIGHT_TOP, Corners3DIndex.BACK_RIGHT_TOP),
        (Corners3DIndex.BACK_RIGHT_TOP, Corners3DIndex.BACK_LEFT_TOP),
        (Corners3DIndex.BACK_LEFT_TOP, Corners3DIndex.FRONT_LEFT_TOP),
        (Corners3DIndex.FRONT_LEFT_BOTTOM, Corners3DIndex.FRONT_LEFT_TOP),
        (Corners3DIndex.FRONT_RIGHT_BOTTOM, Corners3DIndex.FRONT_RIGHT_TOP),
        (Corners3DIndex.BACK_RIGHT_BOTTOM, Corners3DIndex.BACK_RIGHT_TOP),
        (Corners3DIndex.BACK_LEFT_BOTTOM, Corners3DIndex.BACK_LEFT_TOP),
    ]
    lines = np.zeros((len(index_pairs), 2, len(Point3DIndex)), dtype=np.float64)
    for i, (start_idx, end_idx) in enumerate(index_pairs):
        lines[i, 0] = corners[start_idx]
        lines[i, 1] = corners[end_idx]
    return lines


def translate_points_3d(points_3d: npt.NDArray[np.float64], point_3d: Point3D) -> npt.NDArray[np.float64]:
    return points_3d - point_3d.array


def get_bounding_box_outlines(scene: AbstractScene, iteration: int):

    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)

    lines = []
    colors = []
    for box_detection in box_detections:
        bbox: BoundingBoxSE3 = box_detection.bounding_box_se3
        bbox_lines = _get_bounding_box_lines(bbox)
        bbox_lines = translate_points_3d(bbox_lines, initial_ego_vehicle_state.center_se3.point_3d)
        bbox_color = np.zeros(bbox_lines.shape, dtype=np.float32)
        bbox_color[..., :] = (
            BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
            .fill_color.set_brightness(BRIGHTNESS_FACTOR)
            .rgb_norm
        )

        lines.append(bbox_lines)
        colors.append(bbox_color)

    ego_bbox_lines = _get_bounding_box_lines(ego_vehicle_state.bounding_box_se3)
    ego_bbox_lines = translate_points_3d(ego_bbox_lines, initial_ego_vehicle_state.center_se3.point_3d)
    ego_bbox_color = np.zeros(ego_bbox_lines.shape, dtype=np.float32)
    ego_bbox_color[..., :] = EGO_VEHICLE_CONFIG.fill_color.set_brightness(BRIGHTNESS_FACTOR).rgb_norm

    lines.append(ego_bbox_lines)
    colors.append(ego_bbox_color)
    return np.concatenate(lines, axis=0), np.concatenate(colors, axis=0)
