import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import viser

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.utils.display_isp import apply_display_isp

logger = logging.getLogger(__name__)

_IMAGE_SCALE_OPTIONS = ("1", "2", "4", "8")

# Strip slots in display order (left to right). Each slot accepts the first available
# camera id, so both f-theta and pinhole rigs are supported.
_STRIP_SLOTS: Tuple[Tuple[str, Tuple[CameraID, ...]], ...] = (
    ("Front Left", (CameraID.FTCAM_L0, CameraID.PCAM_L0)),
    ("Front", (CameraID.FTCAM_F0, CameraID.PCAM_F0)),
    ("Front Right", (CameraID.FTCAM_R0, CameraID.PCAM_R0)),
)


@dataclass
class CameraStripConfig:
    front_left: bool = False
    front: bool = False
    front_right: bool = False
    image_scale: int = 4


class CameraStripController:
    """Overlays the front-facing cameras as a 1x3 strip on top of the 3D view.

    The strip is a fixed-position HTML overlay: full viewport width minus the control
    panel, images directly adjacent in (front-left, front, front-right) order.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        config: CameraStripConfig,
        context: ElementContext,
        reserved_right_em: float,
    ) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._reserved_right_em = reserved_right_em
        self._html_handle: Optional[viser.GuiHtmlHandle] = None
        self._gui_checkboxes: Dict[str, viser.GuiCheckboxHandle] = {}
        self._gui_image_scale: Optional[viser.GuiDropdownHandle] = None
        self._current_iteration: int = 0

        # Resolve each slot to the first camera id present in the scene.
        available_ids = set(context.scene.get_camera_metadatas())
        self._slot_cameras: Dict[str, Optional[CameraID]] = {
            label: next((cam_id for cam_id in candidates if cam_id in available_ids), None)
            for label, candidates in _STRIP_SLOTS
        }

    def create_gui(self) -> None:
        """Create the Camera Strip folder with per-slot checkboxes and a scale dropdown."""
        if all(cam_id is None for cam_id in self._slot_cameras.values()):
            return

        initial_enabled = {
            "Front Left": self._config.front_left,
            "Front": self._config.front,
            "Front Right": self._config.front_right,
        }
        with self._server.gui.add_folder("Camera Strip", expand_by_default=False):
            for label, _ in _STRIP_SLOTS:
                checkbox = self._server.gui.add_checkbox(
                    label,
                    initial_value=initial_enabled[label] and self._slot_cameras[label] is not None,
                    disabled=self._slot_cameras[label] is None,
                )
                self._gui_checkboxes[label] = checkbox

                @checkbox.on_update
                def _on_checkbox_changed(_) -> None:
                    self._refresh()

            self._gui_image_scale = self._server.gui.add_dropdown(
                "Image Scale",
                _IMAGE_SCALE_OPTIONS,
                initial_value=str(self._config.image_scale)
                if str(self._config.image_scale) in _IMAGE_SCALE_OPTIONS
                else "4",
            )

            @self._gui_image_scale.on_update
            def _on_scale_changed(_) -> None:
                assert self._gui_image_scale is not None
                self._config.image_scale = int(self._gui_image_scale.value)
                self._refresh()

        # The overlay itself lives outside the folder; content is filled on refresh.
        self._html_handle = self._server.gui.add_html("")

    def update(self, iteration: int) -> None:
        """Update the strip for the current iteration."""
        self._current_iteration = iteration
        self._refresh()

    def remove(self) -> None:
        """Clean up handles."""
        self._html_handle = None
        self._gui_checkboxes = {}
        self._gui_image_scale = None

    def _enabled_slots(self) -> List[Tuple[str, CameraID]]:
        enabled = []
        for label, _ in _STRIP_SLOTS:
            checkbox = self._gui_checkboxes.get(label)
            cam_id = self._slot_cameras[label]
            if checkbox is not None and checkbox.value and cam_id is not None:
                enabled.append((label, cam_id))
        return enabled

    def _refresh(self) -> None:
        if self._html_handle is None:
            return

        enabled = self._enabled_slots()
        if len(enabled) == 0:
            if self._html_handle.content != "":
                self._html_handle.content = ""
            return

        image_tags: List[str] = []
        for label, cam_id in enabled:
            camera = self._context.scene.get_camera_at_iteration(
                self._current_iteration, cam_id, scale=self._config.image_scale
            )
            image = camera.rgb_image if camera is not None else None
            if image is None:
                continue
            image = apply_display_isp(image, getattr(camera.metadata, "isp", None))
            encoded = base64.b64encode(iio.imwrite("<bytes>", image, extension=".jpeg")).decode("ascii")
            image_tags.append(
                f'<img alt="{label}" src="data:image/jpeg;base64,{encoded}" '
                'style="width:33.333%;display:block;object-fit:contain;align-self:flex-start;"/>'
            )

        if len(image_tags) == 0:
            if self._html_handle.content != "":
                self._html_handle.content = ""
            return

        self._html_handle.content = (
            '<div style="position:fixed;top:0;left:0;'
            f"width:calc(100vw - {self._reserved_right_em}em);"
            'display:flex;justify-content:center;gap:0;z-index:10;pointer-events:none;">'
            + "".join(image_tags)
            + "</div>"
        )
