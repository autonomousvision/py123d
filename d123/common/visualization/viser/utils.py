from typing import Final, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import trimesh

from d123.common.visualization.color.color import TAB_10, Color
from d123.common.visualization.color.default import BOX_DETECTION_CONFIG, MAP_SURFACE_CONFIG
from d123.datatypes.detections.detection_types import DetectionType
from d123.datatypes.maps.abstract_map import MapLayer
from d123.datatypes.maps.abstract_map_objects import AbstractLane, AbstractSurfaceMapObject
from d123.datatypes.scene.abstract_scene import AbstractScene
from d123.datatypes.sensors.camera import Camera, CameraType
from d123.datatypes.sensors.lidar import LiDARType
from d123.geometry import Corners3DIndex, Point3D, Point3DIndex, Polyline3D, StateSE3, StateSE3Index
from d123.geometry.geometry_index import BoundingBoxSE3Index
from d123.geometry.transform.transform_euler_se3 import convert_relative_to_absolute_points_3d_array
from d123.geometry.transform.transform_se3 import convert_relative_to_absolute_se3_array
from d123.geometry.utils.bounding_box_utils import (
    bbse3_array_to_corners_array,
    corners_array_to_3d_mesh,
)

# TODO: Refactor this file.
# TODO: Add general utilities for 3D primitives and mesh support.

MAP_RADIUS: Final[float] = 200
BRIGHTNESS_FACTOR: Final[float] = 1.0


def configure_trimesh(mesh: trimesh.Trimesh, color: Color):
    # base_color = [r / 255.0 for r in color.rgba]
    mesh.visual.face_colors = color.rgba

    # pbr_material = trimesh.visual.material.PBRMaterial(
    #     baseColorFactor=base_color,  # Your desired color (RGBA, 0-1 range)
    #     metallicFactor=0.0,  # 0.0 = non-metallic (more matte)
    #     roughnessFactor=1.0,  # 0.8 = quite rough (less shiny, 0=mirror, 1=completely rough)
    #     emissiveFactor=[0.0, 0.0, 0.0],  # No emission
    #     alphaCutoff=0.9,  # Alpha threshold for transparency
    #     doubleSided=True,  # Single-sided material
    # )
    # mesh.visual.material = pbr_material
    # mesh.visual = mesh.visual.to_texture()

    return mesh


def get_bounding_box_meshes(scene: AbstractScene, iteration: int):

    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)

    # Load boxes to visualize, including ego vehicle at the last position
    boxes = [bd.bounding_box_se3 for bd in box_detections.box_detections] + [ego_vehicle_state.bounding_box_se3]
    boxes_type = [bd.metadata.detection_type for bd in box_detections.box_detections] + [DetectionType.EGO]

    # create meshes for all boxes
    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_vehicle_state.center_se3.array[StateSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_vertices, box_faces = corners_array_to_3d_mesh(box_corners_array)

    # Create colors for each box based on detection type
    box_colors = []
    for box_type in boxes_type:
        box_colors.append(BOX_DETECTION_CONFIG[box_type].fill_color.rgba)

    # Convert to numpy array and repeat for each vertex
    box_colors = np.array(box_colors)
    vertex_colors = np.repeat(box_colors, 8, axis=0)  # 8 vertices per box

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
    mesh.visual.vertex_colors = vertex_colors

    return mesh


def _get_bounding_box_lines_from_array(corners_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    assert corners_array.shape[-1] == len(Point3DIndex)
    assert corners_array.shape[-2] == len(Corners3DIndex)
    assert corners_array.ndim >= 2

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

    # Handle both single box and batched cases
    if corners_array.ndim == 2:
        # Single box case: (8, 3)
        lines = np.zeros((len(index_pairs), 2, len(Point3DIndex)), dtype=np.float64)
        for i, (start_idx, end_idx) in enumerate(index_pairs):
            lines[i, 0] = corners_array[start_idx]
            lines[i, 1] = corners_array[end_idx]
    else:
        # Batched case: (..., 8, 3)
        batch_shape = corners_array.shape[:-2]
        lines = np.zeros(batch_shape + (len(index_pairs), 2, len(Point3DIndex)), dtype=np.float64)
        for i, (start_idx, end_idx) in enumerate(index_pairs):
            lines[..., i, 0, :] = corners_array[..., start_idx, :]
            lines[..., i, 1, :] = corners_array[..., end_idx, :]

    return lines


def get_bounding_box_outlines(scene: AbstractScene, iteration: int):

    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)

    # Load boxes to visualize, including ego vehicle at the last position
    boxes = [bd.bounding_box_se3 for bd in box_detections.box_detections] + [ego_vehicle_state.bounding_box_se3]
    boxes_type = [bd.metadata.detection_type for bd in box_detections.box_detections] + [DetectionType.EGO]

    # Create lines for all boxes
    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_vehicle_state.center_se3.array[StateSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_lines = _get_bounding_box_lines_from_array(box_corners_array)

    # Create colors for all boxes
    box_colors = np.zeros(box_lines.shape, dtype=np.float32)
    for i, box_type in enumerate(boxes_type):
        box_colors[i, ...] = BOX_DETECTION_CONFIG[box_type].fill_color.set_brightness(BRIGHTNESS_FACTOR).rgb_norm

    box_lines = box_lines.reshape(-1, *box_lines.shape[2:])
    box_colors = box_colors.reshape(-1, *box_colors.shape[2:])

    return box_lines, box_colors


def get_map_meshes(scene: AbstractScene):
    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    center = initial_ego_vehicle_state.center_se3
    map_layers = [
        MapLayer.LANE_GROUP,
        MapLayer.LANE,
        MapLayer.WALKWAY,
        MapLayer.CROSSWALK,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
    ]

    map_objects_dict = scene.map_api.get_proximal_map_objects(center.point_2d, radius=MAP_RADIUS, layers=map_layers)
    output = {}

    for map_layer in map_objects_dict.keys():
        surface_meshes = []
        for map_surface in map_objects_dict[map_layer]:
            map_surface: AbstractSurfaceMapObject
            trimesh_mesh = map_surface.trimesh_mesh
            if map_layer in [
                MapLayer.WALKWAY,
                MapLayer.CROSSWALK,
                MapLayer.GENERIC_DRIVABLE,
                MapLayer.CARPARK,
            ]:
                # Push meshes up by a few centimeters to avoid overlap with the ground in the visualization.
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z - 0.1).array
            else:
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z).array

            if not scene.log_metadata.map_has_z:
                trimesh_mesh.vertices += Point3D(
                    x=0, y=0, z=center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2
                ).array

            trimesh_mesh = configure_trimesh(trimesh_mesh, MAP_SURFACE_CONFIG[map_layer].fill_color)
            surface_meshes.append(trimesh_mesh)
        output[f"{map_layer.serialize()}"] = trimesh.util.concatenate(surface_meshes)
    return output


def get_map_lines(scene: AbstractScene):
    map_layers = [MapLayer.LANE, MapLayer.ROAD_EDGE]
    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    center = initial_ego_vehicle_state.center_se3
    map_objects_dict = scene.map_api.get_proximal_map_objects(center.point_2d, radius=MAP_RADIUS, layers=map_layers)

    def polyline_to_segments(polyline: Polyline3D) -> npt.NDArray[np.float64]:
        polyline_array = polyline.array - center.point_3d.array
        polyline_array = polyline_array.reshape(-1, 1, 3)
        polyline_array = np.concatenate([polyline_array[:-1], polyline_array[1:]], axis=1)
        return polyline_array

    centerlines, right_boundaries, left_boundaries, road_edges = [], [], [], []
    for lane in map_objects_dict[MapLayer.LANE]:
        lane: AbstractLane

        centerlines.append(polyline_to_segments(lane.centerline))
        right_boundaries.append(polyline_to_segments(lane.right_boundary))
        left_boundaries.append(polyline_to_segments(lane.left_boundary))

    for road_edge in map_objects_dict[MapLayer.ROAD_EDGE]:
        road_edges.append(polyline_to_segments(road_edge.polyline_3d))

    centerlines = np.concatenate(centerlines, axis=0)
    left_boundaries = np.concatenate(left_boundaries, axis=0)
    right_boundaries = np.concatenate(right_boundaries, axis=0)
    road_edges = np.concatenate(road_edges, axis=0)

    if not scene.log_metadata.map_has_z:
        # If the map does not have a z-coordinate, we set it to the height of the ego vehicle.
        centerlines[:, :, 2] += center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2
        left_boundaries[:, :, 2] += center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2
        right_boundaries[:, :, 2] += center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2
        road_edges[:, :, 2] += center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2

    return centerlines, left_boundaries, right_boundaries, road_edges


def get_trimesh_from_boundaries(
    left_boundary: Polyline3D, right_boundary: Polyline3D, resolution: float = 1.0
) -> trimesh.Trimesh:
    resolution = 1.0  # [m]

    average_length = (left_boundary.length + right_boundary.length) / 2
    num_samples = int(average_length // resolution) + 1
    left_boundary_array = _interpolate_polyline(left_boundary, num_samples=num_samples)
    right_boundary_array = _interpolate_polyline(right_boundary, num_samples=num_samples)
    return _create_lane_mesh_from_boundary_arrays(left_boundary_array, right_boundary_array)


def _interpolate_polyline(polyline_3d: Polyline3D, num_samples: int = 20) -> npt.NDArray[np.float64]:
    if num_samples < 2:
        num_samples = 2
    distances = np.linspace(0, polyline_3d.length, num=num_samples, endpoint=True, dtype=np.float64)
    return polyline_3d.interpolate(distances)


def _create_lane_mesh_from_boundary_arrays(
    left_boundary_array: npt.NDArray[np.float64],
    right_boundary_array: npt.NDArray[np.float64],
) -> trimesh.Trimesh:

    # Ensure both polylines have the same number of points
    if left_boundary_array.shape[0] != right_boundary_array.shape[0]:
        raise ValueError("Both polylines must have the same number of points")

    n_points = left_boundary_array.shape[0]

    # Combine vertices from both polylines
    vertices = np.vstack([left_boundary_array, right_boundary_array])

    # Create faces by connecting corresponding points on the two polylines
    faces = []
    for i in range(n_points - 1):
        faces.append([i, i + n_points, i + 1])
        faces.append([i + 1, i + n_points, i + n_points + 1])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = MAP_SURFACE_CONFIG[MapLayer.LANE].fill_color.rgba
    return mesh


def get_camera_if_available(scene: AbstractScene, camera_type: CameraType, iteration: int) -> Optional[Camera]:
    camera: Optional[Camera] = None
    if camera_type in scene.available_camera_types:
        camera: Camera = scene.get_camera_at_iteration(iteration, camera_type)
    return camera


def get_camera_values(
    scene: AbstractScene, camera: Camera, iteration: int, resize_factor: Optional[float] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:

    initial_point_3d = scene.get_ego_state_at_iteration(0).center_se3.point_3d
    rear_axle = scene.get_ego_state_at_iteration(iteration).rear_axle_se3

    rear_axle_array = rear_axle.array
    rear_axle_array[:3] -= initial_point_3d.array
    rear_axle = StateSE3.from_array(rear_axle_array, copy=False)

    camera_to_ego = camera.extrinsic  # 4x4 transformation from camera to ego frame
    camera_se3 = StateSE3.from_transformation_matrix(camera_to_ego)

    camera_se3_array = convert_relative_to_absolute_se3_array(origin=rear_axle, se3_array=camera_se3.array)
    abs_camera_se3 = StateSE3.from_array(camera_se3_array, copy=False)

    # Camera transformation in ego frame
    camera_position = abs_camera_se3.point_3d.array
    camera_rotation = abs_camera_se3.quaternion.array

    camera_image = camera.image

    if resize_factor is not None:
        new_width = int(camera_image.shape[1] * resize_factor)
        new_height = int(camera_image.shape[0] * resize_factor)
        camera_image = cv2.resize(camera_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return camera_position, camera_rotation, camera_image


def get_lidar_points(
    scene: AbstractScene, iteration: int, lidar_types: List[LiDARType]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    current_ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)

    def float_to_rgb(values: npt.NDArray[np.float32], cmap_name: str = "viridis") -> npt.NDArray[np.float32]:
        """
        Converts an array of float values to RGB colors using a matplotlib colormap.
        Normalizes values to [0, 1] using min-max scaling.
        Returns an array of shape (N, 3) with RGB values in [0, 1].
        """
        import matplotlib.pyplot as plt

        vmin = np.min(values)
        vmax = np.max(values)
        if vmax > vmin:
            normed = (values - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(values)
        cmap = plt.get_cmap(cmap_name)
        rgb = cmap(normed)[:, :3]  # Ignore alpha channel
        return rgb.astype(np.float32)

    points_ = []
    colors_ = []
    for lidar_idx, lidar_type in enumerate(lidar_types):
        if lidar_type not in scene.available_lidar_types:
            continue
        lidar = scene.get_lidar_at_iteration(iteration, lidar_type)

        # 1. convert points to ego frame
        points = lidar.xyz

        # 2. convert points to world frame
        origin = current_ego_vehicle_state.rear_axle_se3
        points = convert_relative_to_absolute_points_3d_array(origin, points)
        points = points - initial_ego_vehicle_state.center_se3.point_3d.array
        points_.append(points)
        colors_.append([TAB_10[lidar_idx % len(TAB_10)].rgb] * points.shape[0])
        # colors_.append(float_to_rgb(lidar.intensity, cmap_name="viridis"))

    points_ = np.concatenate(points_, axis=0) if points_ else np.empty((0, 3), dtype=np.float32)
    colors_ = np.concatenate(colors_, axis=0) if colors_ else np.empty((0, 3), dtype=np.float32)

    return points_, colors_
