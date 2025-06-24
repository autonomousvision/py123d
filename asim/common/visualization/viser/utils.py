import numpy as np
import numpy.typing as npt
import trimesh

from asim.common.geometry.base import Point3D
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from asim.common.geometry.line.polylines import Polyline3D
from asim.common.visualization.color.config import PlotConfig
from asim.common.visualization.color.default import BOX_DETECTION_CONFIG, EGO_VEHICLE_CONFIG, MAP_SURFACE_CONFIG
from asim.dataset.maps.abstract_map import MapSurfaceType
from asim.dataset.maps.abstract_map_objects import AbstractSurfaceMapObject
from asim.dataset.scene.abstract_scene import AbstractScene

# def bounding_box_to_trimesh(bbox: BoundingBoxSE3, plot_config: PlotConfig) -> trimesh.Trimesh:

#     # Create a unit box centered at origin
#     box_mesh = trimesh.creation.box(extents=[bbox.length, bbox.width, bbox.height])

#     # Create rotation matrix from roll, pitch, yaw (intrinsic rotations)
#     # Using 'xyz' convention: roll (x), pitch (y), yaw (z)
#     rotation = Rotation.from_euler("xyz", [bbox.center.roll, bbox.center.pitch, bbox.center.yaw])
#     rotation_matrix = rotation.as_matrix()

#     # Create 4x4 transformation matrix
#     transform_matrix = np.eye(4)
#     transform_matrix[:3, :3] = rotation_matrix
#     transform_matrix[:3, 3] = [bbox.center.x, bbox.center.y, bbox.center.z]

#     # Apply transformation to the box
#     box_mesh.apply_transform(transform_matrix)

#     base_color = [r / 255.0 for r in plot_config.fill_color.rgba]
#     box_mesh.visual.face_colors = plot_config.fill_color.rgba

#     pbr_material = trimesh.visual.material.PBRMaterial(
#         baseColorFactor=base_color,  # Your desired color (RGBA, 0-1 range)
#         metallicFactor=1.0,  # 0.0 = non-metallic (more matte)
#         roughnessFactor=0.9,  # 0.8 = quite rough (less shiny, 0=mirror, 1=completely rough)
#         emissiveFactor=[0.0, 0.0, 0.0],  # No emission
#         alphaCutoff=0.75,  # Alpha threshold for transparency
#         doubleSided=False,  # Single-sided material
#     )
#     box_mesh.visual.material = pbr_material

#     return box_mesh


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


def get_bounding_box_meshes(scene: AbstractScene, iteration: int, center: Point3D):
    ego_vehicle_state = scene.get_ego_vehicle_state_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)
    # traffic_light_detections = scene.get_traffic_light_detections_at_iteration(iteration)
    # map_api = scene.map_api

    output = {}
    for box_detection in box_detections:
        bbox: BoundingBoxSE3 = box_detection.bounding_box
        bbox = translate_bounding_box_se3(bbox, center)
        plot_config = BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
        trimesh_box = bounding_box_to_trimesh(bbox, plot_config)
        output[f"{box_detection.metadata.detection_type.serialize()}/{box_detection.metadata.track_token}"] = (
            trimesh_box
        )

    ego_bbox = translate_bounding_box_se3(ego_vehicle_state.bounding_box, center)
    trimesh_box = bounding_box_to_trimesh(ego_bbox, EGO_VEHICLE_CONFIG)
    output["ego"] = trimesh_box
    return output


def get_map_meshes(scene: AbstractScene, center: Point3D):
    map_surface_types = [MapSurfaceType.LANE, MapSurfaceType.WALKWAY, MapSurfaceType.CROSSWALK, MapSurfaceType.CARPARK]

    radius = 500
    map_objects_dict = scene.map_api.get_proximal_map_objects(center.point_2d, radius=radius, layers=map_surface_types)
    output = {}

    for map_surface_type in map_objects_dict.keys():
        surface_meshes = []
        for map_surface in map_objects_dict[map_surface_type]:
            map_surface: AbstractSurfaceMapObject
            # outline_line = extract_outline_line(map_surface, center, z=0)
            trimesh_mesh = map_surface.trimesh_mesh
            if map_surface_type in [MapSurfaceType.WALKWAY, MapSurfaceType.CROSSWALK]:
                # trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=1.777 / 2 - 0.05).array
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z + 1.777 / 2 - 0.05).array
            else:
                # trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=1.777 / 2).array
                trimesh_mesh.vertices -= Point3D(x=center.x, y=center.y, z=center.z + 1.777 / 2).array

            if not scene.log_metadata.map_has_z:
                trimesh_mesh.vertices += Point3D(x=0, y=0, z=center.z).array

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
