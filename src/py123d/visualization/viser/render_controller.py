import io
import logging
import zipfile
from typing import Optional

import imageio.v3 as iio
import viser
from tqdm import tqdm

from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.utils.view_utils import get_ego_3rd_person_view_position, get_ego_bev_view_position

logger = logging.getLogger(__name__)


class RenderController:
    """Manages the render-to-file workflow (gif, mp4, png)."""

    def __init__(
        self,
        server: viser.ViserServer,
        context: ElementContext,
        playback_controller: PlaybackController,
    ) -> None:
        self._server = server
        self._context = context
        self._playback = playback_controller
        self._gui_format: Optional[viser.GuiDropdownHandle] = None
        self._gui_view: Optional[viser.GuiDropdownHandle] = None

    def create_gui(self) -> None:
        """Create the Render folder with format, view, and render button."""
        with self._server.gui.add_folder("Render", expand_by_default=False):
            self._gui_format = self._server.gui.add_dropdown("Format", ["gif", "mp4", "png"], initial_value="mp4")
            self._gui_view = self._server.gui.add_dropdown(
                "View", ["3rd Person", "BEV", "Manual"], initial_value="3rd Person"
            )
            render_button = self._server.gui.add_button("Render Scene")
            render_button.on_click(self._on_render)

    def _on_render(self, event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return

        client.scene.reset()
        self._playback.is_rendering = True
        images = []
        scene = self._context.scene
        initial_ego_state = self._context.initial_ego_state

        for i in tqdm(range(scene.number_of_iterations)):
            self._playback.set_timestep(i)
            if self._gui_view.value == "BEV":
                ego_view = get_ego_bev_view_position(scene, i, initial_ego_state)
                client.camera.position = ego_view.point_3d.array
                client.camera.wxyz = ego_view.quaternion.array
            elif self._gui_view.value == "3rd Person":
                ego_view = get_ego_3rd_person_view_position(scene, i, initial_ego_state)
                client.camera.position = ego_view.point_3d.array
                client.camera.wxyz = ego_view.quaternion.array
            images.append(client.get_render(height=1080, width=1920))

        format = self._gui_format.value
        content: bytes
        if format == "gif":
            buffer = io.BytesIO()
            iio.imwrite(buffer, images, extension=".gif", loop=False)
            content = buffer.getvalue()
        elif format == "mp4":
            buffer = io.BytesIO()
            iio.imwrite(buffer, images, extension=".mp4", fps=20)
            content = buffer.getvalue()
        elif format == "png":
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for idx, img in enumerate(images):
                    name = f"frame_{idx:05d}.png"
                    if isinstance(img, (bytes, bytearray)):
                        zf.writestr(name, img)
                    else:
                        img_bytes = io.BytesIO()
                        iio.imwrite(img_bytes, img, extension=".png")
                        zf.writestr(name, img_bytes.getvalue())
            content = zip_buf.getvalue()
            format = "zip"

        scene_name = f"{scene.log_metadata.split}_{scene.scene_uuid}"
        client.send_file_download(f"{scene_name}.{format}", content)
        self._playback.is_rendering = False
