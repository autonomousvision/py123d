import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import viser

from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.utils.display_isp import apply_display_isp

logger = logging.getLogger(__name__)

_IMAGE_SCALE_OPTIONS = ("1", "2", "4", "8")


@dataclass
class CameraGuiConfig:
    visible: bool = False
    image_scale: int = 2
    selected_camera: Optional[CameraID] = None
    selected_modality: str = "camera"


class CameraGuiController:
    """Displays a single selected camera image in its own GUI folder with a dropdown for camera selection."""

    def __init__(self, server: viser.ViserServer, config: CameraGuiConfig, context: ElementContext) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._image_handle: Optional[viser.GuiImageHandle] = None
        self._folder: Optional[viser.GuiFolderHandle] = None
        self._gui_camera_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._gui_modality_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._gui_visible: Optional[viser.GuiCheckboxHandle] = None
        self._current_iteration: int = 0

        # Build camera ID lookup from available cameras
        metadatas = context.scene.get_camera_metadatas()
        self._camera_ids: Dict[str, CameraID] = {cam_id.serialize(lower=False): cam_id for cam_id in metadatas}
        self._camera_names: List[str] = list(self._camera_ids.keys())

        # Map each selectable modality to the scene getter that fetches it (shared signature). Each fetched
        # stream is displayed through Camera.rgb_image, so the GUI renders camera/semantic/instance/depth
        # uniformly (rgb_image colorizes label maps and depth rasters for us).
        self._modality_getters: Dict[str, Callable[..., Optional[Camera]]] = {
            "camera": context.scene.get_camera_at_iteration,
            "semantic": context.scene.get_camera_semantic_at_iteration,
            "instance": context.scene.get_camera_instance_at_iteration,
            "depth": context.scene.get_camera_depth_at_iteration,
        }
        # Only offer a modality whose metadata is present in the scene.
        modality_available: Dict[str, bool] = {
            "camera": len(self._camera_ids) > 0,
            "semantic": len(context.scene.get_camera_semantic_metadatas()) > 0,
            "instance": len(context.scene.get_camera_instance_metadatas()) > 0,
            "depth": len(context.scene.get_camera_depth_metadatas()) > 0,
        }
        self._modality_names: List[str] = [name for name in self._modality_getters if modality_available[name]]

    def create_gui(self) -> None:
        """Create the Camera Image folder with dropdown and image display."""
        if len(self._camera_names) == 0:
            return

        self._folder = self._server.gui.add_folder("Camera Image")
        with self._folder:
            self._gui_visible = self._server.gui.add_checkbox("Visible", self._config.visible)
            self._gui_camera_dropdown = self._server.gui.add_dropdown(
                "Camera",
                self._camera_names,
                initial_value=self._camera_names[0]
                if self._config.selected_camera is None
                else self._config.selected_camera.serialize(lower=False),
            )
            self._gui_modality_dropdown = self._server.gui.add_dropdown(
                "Modality",
                self._modality_names,
                initial_value=self._config.selected_modality
                if self._config.selected_modality in self._modality_names
                else self._modality_names[0],
            )
            self._gui_image_scale = self._server.gui.add_dropdown(
                "Image Scale",
                _IMAGE_SCALE_OPTIONS,
                initial_value=str(self._config.image_scale),
            )

            @self._gui_visible.on_update
            def _on_visible_changed(_) -> None:
                assert self._gui_visible is not None, "GUI must be created before handling visibility change."
                self._config.visible = self._gui_visible.value
                if self._image_handle is not None:
                    self._image_handle.visible = self._gui_visible.value
                else:
                    self._refresh_image()

            @self._gui_camera_dropdown.on_update
            def _on_camera_changed(_) -> None:
                assert self._gui_camera_dropdown is not None, "GUI must be created before handling camera change."
                # self._image_handle = None
                self._config.selected_camera = self._camera_ids[self._gui_camera_dropdown.value]
                self._refresh_image()

            @self._gui_modality_dropdown.on_update
            def _on_modality_changed(_) -> None:
                assert self._gui_modality_dropdown is not None, "GUI must be created before handling modality change."
                self._config.selected_modality = self._gui_modality_dropdown.value
                self._refresh_image()

            @self._gui_image_scale.on_update
            def _on_scale_changed(_) -> None:
                assert self._gui_image_scale is not None, "GUI must be created before handling scale change."
                self._config.image_scale = int(self._gui_image_scale.value)
                # self._image_handle = None
                self._refresh_image()

    def update(self, iteration: int) -> None:
        """Update the displayed image for the current iteration."""
        self._current_iteration = iteration
        if self._gui_visible is None or not self._gui_visible.value:
            return
        self._refresh_image()

    def remove(self) -> None:
        """Clean up handles."""
        self._image_handle = None

    def _refresh_image(self) -> None:
        """Fetch and display the image for the currently selected camera."""
        if self._gui_camera_dropdown is None or self._gui_visible is None or self._folder is None:
            return
        if not self._gui_visible.value:
            return

        camera_name = self._gui_camera_dropdown.value
        camera_id = self._camera_ids.get(camera_name)
        if camera_id is None:
            return

        getter = self._modality_getters.get(self._config.selected_modality)
        camera = (
            getter(self._current_iteration, camera_id, scale=self._config.image_scale) if getter is not None else None
        )
        image = camera.rgb_image if camera is not None else None
        if image is not None and camera is not None:
            # Only true camera streams carry a display-ISP block; label/depth maps do not.
            image = apply_display_isp(image, getattr(camera.metadata, "isp", None))
        if image is None:
            # The selected camera/modality pair is unavailable: hide any stale frame.
            if self._image_handle is not None:
                self._image_handle.visible = False
            return

        if self._image_handle is not None:
            self._image_handle.image = image
            self._image_handle.visible = self._gui_visible.value
        else:
            with self._folder:
                self._image_handle = self._server.gui.add_image(
                    image=image,
                    label=camera_name,
                )
