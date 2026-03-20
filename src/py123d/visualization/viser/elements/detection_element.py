import logging
from typing import Dict, List, Optional, Union

import numpy as np
import trimesh
import trimesh.visual.material
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Corners3DIndex, PoseSE3Index
from py123d.geometry.utils.bounding_box_utils import (
    bbse3_array_to_corners_array,
    corners_array_to_3d_mesh,
    corners_array_to_edge_lines,
)
from py123d.visualization.color.default import BOX_DETECTION_CONFIG
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.viser_config import DetectionConfig

logger = logging.getLogger(__name__)


class DetectionElement(ViewerElement):
    """Visualizes 3D bounding box detections in the scene."""

    def __init__(self, context: ElementContext, config: DetectionConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[str, Optional[Union[viser.GlbHandle, viser.LineSegmentsHandle]]] = {
            "mesh": None,
            "lines": None,
        }
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_type: Optional[viser.GuiDropdownHandle] = None
        self._gui_opacity: Optional[viser.GuiSliderHandle] = None
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Bounding Boxes"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_type = server.gui.add_dropdown(
            "Type", ("mesh", "lines", "mesh+lines"), initial_value=self._config.type
        )
        self._gui_opacity = server.gui.add_slider(
            "Opacity", min=0.0, max=1.0, step=0.05, initial_value=self._config.opacity
        )
        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_type.on_update(self._on_type_changed)
        self._gui_opacity.on_update(self._on_opacity_changed)

    def update(self, iteration: int) -> None:
        self._current_iteration = iteration
        visible_handle_keys: List[str] = []
        display_type = self._gui_type.value

        if self._gui_visible.value:
            if display_type in {"mesh", "mesh+lines"}:
                opacity = self._gui_opacity.value
                mesh = _get_bounding_box_meshes(
                    self._context.scene, iteration, self._context.initial_ego_state, opacity=opacity
                )
                self._handles["mesh"] = self._server.scene.add_mesh_trimesh(
                    "box_detections_mesh",
                    mesh=mesh,
                    visible=True,
                    cast_shadow=False,
                )
                visible_handle_keys.append("mesh")
            if display_type in {"lines", "mesh+lines"}:
                lines, colors, _ = _get_bounding_box_outlines(
                    self._context.scene, iteration, self._context.initial_ego_state
                )
                self._handles["lines"] = self._server.scene.add_line_segments(
                    "box_detections_lines",
                    points=lines,
                    colors=colors,
                    line_width=self._config.line_width,
                    visible=True,
                )
                visible_handle_keys.append("lines")

        for key in self._handles:
            if key not in visible_handle_keys and self._handles[key] is not None:
                self._handles[key].visible = False

    def remove(self) -> None:
        for handle in self._handles.values():
            if handle is not None:
                handle.remove()
        self._handles = {"mesh": None, "lines": None}

    def _on_visibility_changed(self, _) -> None:
        if self._gui_visible.value:
            self.update(self._current_iteration)
        else:
            for handle in self._handles.values():
                if handle is not None:
                    handle.visible = False

    def _on_type_changed(self, _) -> None:
        self._config.type = self._gui_type.value
        self.update(self._current_iteration)

    def _on_opacity_changed(self, _) -> None:
        self._config.opacity = self._gui_opacity.value
        self.update(self._current_iteration)


def _get_bounding_box_meshes(
    scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3, opacity: float = 1.0
) -> trimesh.Trimesh:
    ego_vehicle_state = scene.get_ego_state_se3_at_iteration(iteration)
    box_detections = scene.get_box_detections_se3_at_iteration(iteration)

    if box_detections is None:
        box_detections_list = []
    else:
        box_detections_list = box_detections.box_detections

    boxes = [bd.bounding_box_se3 for bd in box_detections_list] + [ego_vehicle_state.bounding_box_se3]
    boxes_labels = [bd.attributes.default_label for bd in box_detections_list] + [DefaultBoxDetectionLabel.EGO]

    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_vertices, box_faces = corners_array_to_3d_mesh(box_corners_array)

    alpha = int(np.clip(opacity * 255, 0, 255))
    box_colors = []
    for box_label in boxes_labels:
        r, g, b, _ = BOX_DETECTION_CONFIG[box_label].fill_color.rgba
        box_colors.append((r, g, b, alpha))

    box_colors = np.array(box_colors)
    vertex_colors = np.repeat(box_colors, len(Corners3DIndex), axis=0)

    mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
    mesh.visual.vertex_colors = vertex_colors
    mesh.visual.material = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")

    return mesh


def _get_bounding_box_outlines(scene: SceneAPI, iteration: int, initial_ego_state: EgoStateSE3) -> tuple:
    ego_vehicle_state = scene.get_ego_state_se3_at_iteration(iteration)
    box_detections = scene.get_box_detections_se3_at_iteration(iteration)

    box_detections_list = box_detections.box_detections if box_detections is not None else []
    boxes = [bd.bounding_box_se3 for bd in box_detections_list] + [ego_vehicle_state.bounding_box_se3]
    boxes_labels = [bd.attributes.default_label for bd in box_detections_list] + [DefaultBoxDetectionLabel.EGO]

    box_se3_array = np.array([box.array for box in boxes])
    box_se3_array[..., BoundingBoxSE3Index.XYZ] -= initial_ego_state.center_se3.array[PoseSE3Index.XYZ]
    box_corners_array = bbse3_array_to_corners_array(box_se3_array)
    box_outlines = corners_array_to_edge_lines(box_corners_array)

    box_colors = np.zeros(box_outlines.shape, dtype=np.float32)
    for i, box_label in enumerate(boxes_labels):
        box_colors[i, ...] = BOX_DETECTION_CONFIG[box_label].fill_color.rgb_norm

    box_outlines = box_outlines.reshape(-1, *box_outlines.shape[2:])
    box_colors = box_colors.reshape(-1, *box_colors.shape[2:])

    return box_outlines, box_colors, box_se3_array
