from typing import Dict

import trimesh
import viser

from d123.common.visualization.color.default import MAP_SURFACE_CONFIG
from d123.common.visualization.viser.viser_config import ViserConfig
from d123.datatypes.maps.abstract_map import MapLayer
from d123.datatypes.maps.abstract_map_objects import AbstractSurfaceMapObject
from d123.datatypes.scene.abstract_scene import AbstractScene
from d123.datatypes.vehicle_state.ego_state import EgoStateSE3
from d123.geometry import Point3D, Point3DIndex


def add_map_to_viser_server(
    scene: AbstractScene,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
) -> None:

    if viser_config.map_visible:
        for name, mesh in _get_map_trimesh_dict(scene, initial_ego_state, viser_config).items():
            viser_server.scene.add_mesh_trimesh(f"/map/{name}", mesh, visible=True)


def _get_map_trimesh_dict(
    scene: AbstractScene,
    initial_ego_state: EgoStateSE3,
    viser_config: ViserConfig,
) -> Dict[str, trimesh.Trimesh]:

    # Unpack scene center for translation of map objects.
    scene_center: Point3D = initial_ego_state.center.point_3d
    scene_center_array = scene_center.array

    # Load map objects within a certain radius around the scene center.
    map_layers = [
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.CROSSWALK,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
    ]
    map_objects_dict = scene.get_map_api().get_proximal_map_objects(
        scene_center.point_2d,
        radius=viser_config.map_radius,
        layers=map_layers,
    )

    # Create trimesh meshes for each map layer.
    trimesh_dict = {}
    for map_layer in map_objects_dict.keys():
        surface_meshes = []
        for map_surface in map_objects_dict[map_layer]:
            map_surface: AbstractSurfaceMapObject

            trimesh_mesh = map_surface.trimesh_mesh
            trimesh_mesh.vertices -= scene_center_array

            # Adjust height of non-road surfaces to avoid z-fighting in the visualization.
            if map_layer in [
                MapLayer.WALKWAY,
                MapLayer.CROSSWALK,
                MapLayer.CARPARK,
            ]:
                trimesh_mesh.vertices[..., Point3DIndex.Z] += viser_config.map_non_road_z_offset

            # If the map does not have z-values, we place the surfaces on the ground level of the ego vehicle.
            if not scene.log_metadata.map_has_z:
                trimesh_mesh.vertices[..., Point3DIndex.Z] += (
                    scene_center.z - initial_ego_state.vehicle_parameters.height / 2
                )

            # Color the mesh based on the map layer type.
            trimesh_mesh.visual.face_colors = MAP_SURFACE_CONFIG[map_layer].fill_color.rgba
            surface_meshes.append(trimesh_mesh)

        trimesh_dict[f"{map_layer.serialize()}"] = trimesh.util.concatenate(surface_meshes)

    return trimesh_dict
