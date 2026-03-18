import logging
from typing import Dict, Optional

import numpy as np
import trimesh
import viser

from py123d.api import SceneAPI
from py123d.datatypes.map_objects.base_map_objects import BaseMapSurfaceObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer, StopZoneType
from py123d.datatypes.map_objects.map_objects import StopZone
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import Point3D, Point3DIndex
from py123d.visualization.color.default import MAP_SURFACE_CONFIG
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.viser_config import MapConfig

logger = logging.getLogger(__name__)


class MapElement(ViewerElement):
    """Visualizes map layers (lanes, crosswalks, etc.) in the 3D scene."""

    def __init__(self, context: ElementContext, config: MapConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[MapLayer, viser.MeshHandle] = {}
        self._last_query_position: Optional[Point3D] = None
        self._force_update: bool = False
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_radius: Optional[viser.GuiSliderHandle] = None

    @property
    def name(self) -> str:
        return "Map"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_radius = server.gui.add_slider(
            "Radius", min=10.0, max=1000.0, step=1.0, initial_value=self._config.radius
        )
        gui_radius_options = server.gui.add_button_group("Radius Options.", ("25", "50", "100", "500"))

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_radius.on_update(self._on_radius_changed)
        gui_radius_options.on_click(self._on_radius_preset_clicked)

    def update(self, iteration: int) -> None:
        if not self._gui_visible.value:
            return

        map_trimesh_dict: Optional[Dict[MapLayer, trimesh.Trimesh]] = None

        if len(self._handles) == 0 or self._force_update:
            current_ego_state = self._context.initial_ego_state
            map_trimesh_dict = _get_map_trimesh_dict(
                self._context.scene,
                self._context.initial_ego_state,
                current_ego_state,
                self._config.radius,
                self._config.non_road_z_offset,
            )
            self._last_query_position = current_ego_state.center_se3.point_3d
            self._force_update = False

        elif self._config.requery:
            current_ego_state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
            current_position = current_ego_state.center_se3.point_3d

            if np.linalg.norm(current_position.array - self._last_query_position.array) > self._config.radius / 2:
                self._last_query_position = current_position
                map_trimesh_dict = _get_map_trimesh_dict(
                    self._context.scene,
                    self._context.initial_ego_state,
                    current_ego_state,
                    self._config.radius,
                    self._config.non_road_z_offset,
                )

        if map_trimesh_dict is not None:
            for map_layer, mesh in map_trimesh_dict.items():
                self._handles[map_layer] = self._server.scene.add_mesh_trimesh(
                    f"/map/{map_layer.serialize()}",
                    mesh,
                    visible=True,
                )

    def remove(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._last_query_position = None

    def _on_visibility_changed(self, _) -> None:
        for handle in self._handles.values():
            handle.visible = self._gui_visible.value

    def _on_radius_changed(self, _) -> None:
        self._config.radius = self._gui_radius.value
        self._force_update = True

    def _on_radius_preset_clicked(self, event) -> None:
        self._gui_radius.value = float(event.target.value)
        self._force_update = True


def _get_map_trimesh_dict(
    scene: SceneAPI,
    initial_ego_state: EgoStateSE3,
    current_ego_state: EgoStateSE3,
    radius: float,
    non_road_z_offset: float,
) -> Dict[MapLayer, trimesh.Trimesh]:
    output_trimesh_dict: Dict[MapLayer, trimesh.Trimesh] = {}

    scene_center: Point3D = initial_ego_state.center_se3.point_3d
    scene_center_array = scene_center.array
    scene_query_position = current_ego_state.center_se3.point_3d

    map_layers = [
        MapLayer.LANE,
        MapLayer.LANE_GROUP,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.CROSSWALK,
        MapLayer.CARPARK,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.STOP_ZONE,
    ]
    map_api = scene.get_map_api()
    if map_api is not None:
        map_objects_dict = map_api.get_map_objects_in_radius(
            scene_query_position.point_2d,
            radius=radius,
            layers=map_layers,
        )

        if len(map_objects_dict[MapLayer.LANE_GROUP]) == 0:
            map_objects_dict.pop(MapLayer.LANE_GROUP)
        else:
            map_objects_dict.pop(MapLayer.LANE)

        for map_layer in map_objects_dict.keys():
            surface_meshes = []
            for map_surface in map_objects_dict[map_layer]:
                map_surface: BaseMapSurfaceObject

                if isinstance(map_surface, StopZone) and map_surface.stop_zone_type == StopZoneType.TURN_STOP:
                    continue

                trimesh_mesh = map_surface.trimesh_mesh
                trimesh_mesh.vertices -= scene_center_array

                if map_layer in {MapLayer.WALKWAY, MapLayer.CROSSWALK, MapLayer.CARPARK, MapLayer.STOP_ZONE}:
                    trimesh_mesh.vertices[..., Point3DIndex.Z] += non_road_z_offset

                if not map_api.map_metadata.map_has_z:
                    trimesh_mesh.vertices[..., Point3DIndex.Z] += (
                        scene_query_position.z - initial_ego_state.metadata.height / 2
                    )

                trimesh_mesh.visual.face_colors = MAP_SURFACE_CONFIG[map_layer].fill_color.rgba
                surface_meshes.append(trimesh_mesh)

            output_trimesh_dict[map_layer] = trimesh.util.concatenate(surface_meshes)

    return output_trimesh_dict
