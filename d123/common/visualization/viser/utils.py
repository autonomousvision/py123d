from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import trimesh
from pyquaternion import Quaternion
from typing_extensions import Final

# from d123.common.datatypes.sensor.camera_parameters import get_nuplan_camera_parameters
from d123.common.datatypes.sensor.camera import Camera, CameraType
from d123.common.datatypes.sensor.lidar import LiDARType
from d123.common.visualization.color.color import TAB_10, Color
from d123.common.visualization.color.config import PlotConfig
from d123.common.visualization.color.default import BOX_DETECTION_CONFIG, EGO_VEHICLE_CONFIG, MAP_SURFACE_CONFIG
from d123.datasets.maps.abstract_map import MapLayer
from d123.datasets.maps.abstract_map_objects import AbstractLane, AbstractSurfaceMapObject
from d123.datasets.scene.abstract_scene import AbstractScene
from d123.geometry import BoundingBoxSE3, EulerStateSE3, Point3D, Polyline3D
from d123.geometry.transform.transform_euler_se3 import convert_relative_to_absolute_points_3d_array

# TODO: Refactor this file.
# TODO: Add general utilities for 3D primitives and mesh support.

MAP_RADIUS: Final[float] = 500
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


def bounding_box_to_trimesh(bbox: BoundingBoxSE3, plot_config: PlotConfig) -> trimesh.Trimesh:

    # Create a unit box centered at origin
    box_mesh = trimesh.creation.box(extents=[bbox.length, bbox.width, bbox.height])

    # Apply rotations in order: roll, pitch, yaw
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.yaw, [0, 0, 1]))
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.pitch, [0, 1, 0]))
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.roll, [1, 0, 0]))

    # Apply translation
    box_mesh = box_mesh.apply_translation([bbox.center.x, bbox.center.y, bbox.center.z])

    return configure_trimesh(box_mesh, plot_config.fill_color)


def translate_bounding_box_se3(bounding_box_se3: BoundingBoxSE3, point_3d: Point3D) -> BoundingBoxSE3:
    bounding_box_se3.center.x = bounding_box_se3.center.x - point_3d.x
    bounding_box_se3.center.y = bounding_box_se3.center.y - point_3d.y
    bounding_box_se3.center.z = bounding_box_se3.center.z - point_3d.z
    return bounding_box_se3


def get_bounding_box_meshes(scene: AbstractScene, iteration: int):
    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)

    ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)
    # traffic_light_detections = scene.get_traffic_light_detections_at_iteration(iteration)
    # map_api = scene.map_api

    output = {}
    for box_detection in box_detections:
        bbox: BoundingBoxSE3 = box_detection.bounding_box
        bbox = translate_bounding_box_se3(bbox, initial_ego_vehicle_state.center_se3)
        plot_config = BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
        trimesh_box = bounding_box_to_trimesh(bbox, plot_config)
        output[f"{box_detection.metadata.detection_type.serialize()}/{box_detection.metadata.track_token}"] = (
            trimesh_box
        )

    ego_bbox = translate_bounding_box_se3(ego_vehicle_state.bounding_box, initial_ego_vehicle_state.center_se3)
    trimesh_box = bounding_box_to_trimesh(ego_bbox, EGO_VEHICLE_CONFIG)
    output["ego"] = trimesh_box
    return output


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
    print(map_objects_dict.keys())
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


def get_camera_values(scene: AbstractScene, camera: Camera, iteration: int) -> Tuple[Point3D, Quaternion]:
    initial_point_3d = scene.get_ego_state_at_iteration(0).center_se3.point_3d
    rear_axle = scene.get_ego_state_at_iteration(iteration).rear_axle_se3

    rear_axle_array = rear_axle.array
    rear_axle_array[:3] -= initial_point_3d.array
    rear_axle = EulerStateSE3.from_array(rear_axle_array)

    camera_to_ego = camera.extrinsic  # 4x4 transformation from camera to ego frame

    ego_transform = rear_axle.transformation_matrix

    camera_transform = ego_transform @ camera_to_ego

    # Camera transformation in ego frame
    camera_position = Point3D(*camera_transform[:3, 3])
    camera_rotation = Quaternion(matrix=camera_transform[:3, :3])

    return camera_position, camera_rotation


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
