import logging
from typing import Dict, Optional

import viser

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.viser_config import CameraGuiConfig

logger = logging.getLogger(__name__)

_IMAGE_SCALE_OPTIONS = ("1", "2", "4", "8")


class CameraGuiElement(ViewerElement):
    """Displays camera images as embedded GUI panels."""

    def __init__(self, context: ElementContext, config: CameraGuiConfig) -> None:
        self._context = context
        self._config = config
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[CameraID, viser.GuiImageHandle] = {}
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._current_iteration: int = 0

    @property
    def name(self) -> str:
        return "Camera Images"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._gui_visible = server.gui.add_checkbox("Visible", self._config.visible)
        self._gui_image_scale = server.gui.add_dropdown(
            "Image Scale",
            _IMAGE_SCALE_OPTIONS,
            initial_value=str(self._config.image_scale),
        )

        self._gui_visible.on_update(self._on_visibility_changed)
        self._gui_image_scale.on_update(self._on_image_scale_changed)

    def update(self, iteration: int) -> None:
        self._current_iteration = iteration
        if not self._gui_visible.value:
            return

        for camera_type in self._config.types:
            camera = self._context.scene.get_camera_at_iteration(iteration, camera_type, scale=self._config.image_scale)
            if camera is None:
                continue

            if camera_type in self._handles:
                self._handles[camera_type].image = camera.image
            else:
                with self._server.gui.add_folder(f"Camera {camera_type.serialize()}"):
                    self._handles[camera_type] = self._server.gui.add_image(
                        image=camera.image, label=camera_type.serialize()
                    )

    def remove(self) -> None:
        self._handles.clear()

    def _on_visibility_changed(self, _) -> None:
        for handle in self._handles.values():
            handle.visible = self._gui_visible.value

    def _on_image_scale_changed(self, _) -> None:
        self._config.image_scale = int(self._gui_image_scale.value)
        # Clear existing handles so they get recreated at new scale
        self._handles.clear()
        self.update(self._current_iteration)
