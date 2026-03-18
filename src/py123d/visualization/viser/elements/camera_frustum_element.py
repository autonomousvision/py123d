import concurrent.futures
import logging
from typing import Dict, List, Optional

import viser

from py123d.datatypes.sensors.base_camera import CameraID, CameraModel
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.utils.view_utils import decompose_camera_pose, get_scene_center_pose
from py123d.visualization.viser.viser_config import CameraFrustumConfig

logger = logging.getLogger(__name__)


class CameraFrustumElement(ViewerElement):
    """Visualizes camera frustums (pinhole and fisheye) in the 3D scene."""

    def __init__(self, context: ElementContext, config: CameraFrustumConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[CameraID, viser.CameraFrustumHandle] = {}
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_scale: Optional[viser.GuiInputHandle] = None
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._all_types: List[CameraID] = list(config.pinhole_types) + list(config.fisheye_types)
        self._fisheye_set = set(config.fisheye_types)
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Cameras"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_scale = server.gui.add_slider(
            "Frustum Scale",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=self._config.scale,
        )
        self._gui_image_scale = server.gui.add_dropdown(
            "Image Scale",
            ("1", "2", "4", "8"),
            initial_value=str(self._config.image_scale),
        )
        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_scale.on_update(self._on_scale_changed)
        self._gui_image_scale.on_update(self._on_image_scale_changed)

    def update(self, iteration: int) -> None:
        self._current_iteration = iteration
        if not self._gui_visible.value:
            return

        scene_center_pose = get_scene_center_pose(self._context.scene_center_array)

        def _update_frustum(camera_type: CameraID) -> None:
            camera = self._context.scene.get_camera_at_iteration(iteration, camera_type, scale=self._config.image_scale)
            if camera is None:
                return

            camera_position, camera_quaternion = decompose_camera_pose(camera, scene_center_pose)

            # Determine FOV based on camera model
            is_fisheye = camera.metadata.camera_model == CameraModel.FISHEYE_MEI
            fov = self._config.fisheye_fov if is_fisheye else camera.metadata.fov_y

            if camera_type in self._handles:
                self._handles[camera_type].position = camera_position
                self._handles[camera_type].wxyz = camera_quaternion
                self._handles[camera_type].image = camera.image
            else:
                self._handles[camera_type] = self._server.scene.add_camera_frustum(
                    f"camera_frustums/{camera_type.serialize()}",
                    fov=fov,
                    aspect=camera.metadata.aspect_ratio,
                    scale=self._config.scale,
                    image=camera.image,
                    position=camera_position,
                    wxyz=camera_quaternion,
                )

        if len(self._all_types) > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._all_types)) as executor:
                futures = {executor.submit(_update_frustum, ct): ct for ct in self._all_types}
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    def remove(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def _on_visibility_changed(self, _) -> None:
        for handle in self._handles.values():
            handle.visible = self._gui_visible.value

    def _on_scale_changed(self, _) -> None:
        self._config.scale = self._gui_scale.value
        # Frustum scale can't be updated in-place; remove and re-create
        self.remove()
        self.update(self._current_iteration)

    def _on_image_scale_changed(self, _) -> None:
        self._config.image_scale = int(self._gui_image_scale.value)
        # Remove and re-create to reload images at new resolution
        self.remove()
        self.update(self._current_iteration)
