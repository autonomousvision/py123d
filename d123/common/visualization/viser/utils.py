from typing import Tuple

import numpy as np
import numpy.typing as npt
import trimesh
from pyquaternion import Quaternion

# from d123.common.datatypes.sensor.camera_parameters import get_nuplan_camera_parameters
from d123.common.datatypes.sensor.camera import Camera, CameraType
from d123.common.geometry.base import Point3D, StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from d123.common.geometry.line.polylines import Polyline3D
from d123.common.geometry.transform.se3 import convert_relative_to_absolute_points_3d_array
from d123.common.visualization.color.config import PlotConfig
from d123.common.visualization.color.default import BOX_DETECTION_CONFIG, EGO_VEHICLE_CONFIG, MAP_SURFACE_CONFIG
from d123.dataset.maps.abstract_map import MapSurfaceType
from d123.dataset.maps.abstract_map_objects import AbstractSurfaceMapObject
from d123.dataset.scene.abstract_scene import AbstractScene

# TODO: Refactor this file.
# TODO: Add general utilities for 3D primitives and mesh support.


def bounding_box_to_trimesh(bbox: BoundingBoxSE3, plot_config: PlotConfig) -> trimesh.Trimesh:

    # Create a unit box centered at origin
    box_mesh = trimesh.creation.box(extents=[bbox.length, bbox.width, bbox.height])

    # Apply rotations in order: roll, pitch, yaw
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.roll, [1, 0, 0]))
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.pitch, [0, 1, 0]))
    box_mesh = box_mesh.apply_transform(trimesh.transformations.rotation_matrix(bbox.center.yaw, [0, 0, 1]))

    # Apply translation
    box_mesh = box_mesh.apply_translation([bbox.center.x, bbox.center.y, bbox.center.z])
    base_color = [r / 255.0 for r in plot_config.fill_color.rgba]
    box_mesh.visual.face_colors = plot_config.fill_color.rgba

    pbr_material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=base_color,  # Your desired color (RGBA, 0-1 range)
        metallicFactor=1.0,  # 0.0 = non-metallic (more matte)
        roughnessFactor=0.9,  # 0.8 = quite rough (less shiny, 0=mirror, 1=completely rough)
        emissiveFactor=[0.0, 0.0, 0.0],  # No emission
        alphaCutoff=0.75,  # Alpha threshold for transparency
        doubleSided=False,  # Single-sided material
    )
    box_mesh.visual.material = pbr_material

    return box_mesh


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
    map_surface_types = [
        MapSurfaceType.LANE_GROUP,
        MapSurfaceType.WALKWAY,
        MapSurfaceType.CROSSWALK,
        MapSurfaceType.CARPARK,
    ]

    radius = 500
    map_objects_dict = scene.map_api.get_proximal_map_objects(center.point_2d, radius=radius, layers=map_surface_types)
    output = {}

    for map_surface_type in map_objects_dict.keys():
        surface_meshes = []
        for map_surface in map_objects_dict[map_surface_type]:
            map_surface: AbstractSurfaceMapObject
            trimesh_mesh = map_surface.trimesh_mesh
            if map_surface_type in [MapSurfaceType.WALKWAY, MapSurfaceType.CROSSWALK]:
                # trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=1.777 / 2 - 0.05).array
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z - 0.05).array
            else:
                # trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=1.777 / 2).array
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z).array

            if not scene.log_metadata.map_has_z:
                trimesh_mesh.vertices += Point3D(
                    x=0, y=0, z=center.z - initial_ego_vehicle_state.vehicle_parameters.height / 2
                ).array

            trimesh_mesh.visual.face_colors = MAP_SURFACE_CONFIG[map_surface_type].fill_color.rgba
            surface_meshes.append(trimesh_mesh)
        output[f"{map_surface_type.serialize()}"] = trimesh.util.concatenate(surface_meshes)

    return output


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
    mesh.visual.face_colors = MAP_SURFACE_CONFIG[MapSurfaceType.LANE].fill_color.rgba
    return mesh


def get_camera_values(
    scene: AbstractScene, camera_type: CameraType, iteration: int
) -> Tuple[Point3D, Quaternion, Camera]:
    initial_point_3d = scene.get_ego_state_at_iteration(0).center_se3.point_3d
    rear_axle = scene.get_ego_state_at_iteration(iteration).rear_axle_se3

    camera = scene.get_camera_at_iteration(iteration, camera_type)

    rear_axle_array = rear_axle.array
    rear_axle_array[:3] -= initial_point_3d.array
    rear_axle = StateSE3.from_array(rear_axle_array)

    camera_to_ego = camera.extrinsic  # 4x4 transformation from camera to ego frame
    camera.image

    # Get the rotation matrix of the rear axle pose
    from d123.common.geometry.transform.se3 import get_rotation_matrix

    ego_transform = np.eye(4, dtype=np.float64)
    ego_transform[:3, :3] = get_rotation_matrix(rear_axle)
    ego_transform[:3, 3] = rear_axle.point_3d.array

    CARLA_DEBUG = True
    if CARLA_DEBUG:
        rotation = StateSE3(
            0.0,
            0.0,
            0.0,
            -np.deg2rad(90.0),
            np.deg2rad(0.0),
            -np.deg2rad(90.0),
        )
        camera_to_ego[:3, :3] = get_rotation_matrix(rotation)
        camera_transform = ego_transform @ camera_to_ego
    else:
        camera_transform = ego_transform @ camera_to_ego

    # Camera transformation in ego frame

    camera_position = Point3D(*camera_transform[:3, 3])
    camera_rotation = Quaternion(matrix=camera_transform[:3, :3])

    return camera_position, camera_rotation, camera


def _get_ego_frame_pose(scene: AbstractScene, iteration: int) -> StateSE3:

    initial_point_3d = scene.get_ego_state_at_iteration(0).center_se3.point_3d
    state_se3 = scene.get_ego_state_at_iteration(iteration).center_se3

    state_se3.x = state_se3.x - initial_point_3d.x
    state_se3.y = state_se3.y - initial_point_3d.y
    state_se3.z = state_se3.z - initial_point_3d.z

    return state_se3


def euler_to_quaternion_scipy(roll: float, pitch: float, yaw: float) -> npt.NDArray[np.float64]:
    from scipy.spatial.transform import Rotation

    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    quat = r.as_quat(scalar_first=True)
    return quat


def get_lidar_points(scene: AbstractScene, iteration: int) -> npt.NDArray[np.float32]:

    pass

    initial_ego_vehicle_state = scene.get_ego_state_at_iteration(0)
    current_ego_vehicle_state = scene.get_ego_state_at_iteration(iteration)

    lidar = scene.get_lidar_at_iteration(iteration)
    # if scene.log_metadata.dataset == "nuplan":
    # NOTE: nuPlan uses the rear axle as origin.
    origin = current_ego_vehicle_state.rear_axle_se3
    points = lidar.xyz
    points = convert_relative_to_absolute_points_3d_array(origin, points)
    points = points - initial_ego_vehicle_state.center_se3.point_3d.array

    return points
