import io
import logging
import zipfile
from dataclasses import dataclass
from typing import Literal, Optional

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import viser
from tqdm import tqdm

from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.utils.view_utils import get_ego_3rd_person_view_position, get_ego_bev_view_position

logger = logging.getLogger(__name__)


# Matches the viewer canvas background: Mantine dark[9] in dark mode, white otherwise.
DARK_BACKGROUND = (20, 21, 23)
LIGHT_BACKGROUND = (255, 255, 255)

RESOLUTION_MAP = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4K": (3840, 2160),
}


@dataclass
class RenderConfig:
    format: Literal["gif", "mp4", "png"] = "mp4"
    view: Literal["3rd Person", "BEV", "Manual"] = "3rd Person"
    resolution: Literal["480p", "720p", "1080p", "1440p", "4K"] = "1080p"
    fps: Literal["5 fps", "10 fps", "15 fps", "20 fps", "30 fps"] = "20 fps"


class RenderController:
    """Manages the render-to-file workflow (gif, mp4, png)."""

    def __init__(
        self,
        server: viser.ViserServer,
        config: RenderConfig,
        context: ElementContext,
        playback_controller: PlaybackController,
    ) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._playback = playback_controller
        self._gui_format: Optional[viser.GuiDropdownHandle] = None
        self._gui_view: Optional[viser.GuiDropdownHandle] = None
        self._gui_start_frame: Optional[viser.GuiSliderHandle] = None
        self._gui_end_frame: Optional[viser.GuiSliderHandle] = None
        self._gui_background: Optional[viser.GuiRgbHandle] = None

    def create_gui(self) -> None:
        """Create the Render folder with format, view, and render button."""
        with self._server.gui.add_folder("Render", expand_by_default=False):
            self._gui_format = self._server.gui.add_dropdown(
                "Format", ["gif", "mp4", "png"], initial_value=self._config.format
            )
            self._gui_view = self._server.gui.add_dropdown(
                "View", ["3rd Person", "BEV", "Manual"], initial_value=self._config.view
            )
            self._resolution = self._server.gui.add_dropdown(
                "Resolution", ["480p", "720p", "1080p", "1440p", "4K"], initial_value=self._config.resolution
            )
            self._fps = self._server.gui.add_dropdown(
                "FPS", ["5 fps", "10 fps", "15 fps", "20 fps", "30 fps"], initial_value=self._config.fps
            )
            num_frames = self._context.num_frames
            self._gui_start_frame = self._server.gui.add_slider(
                "Start Frame", min=0, max=num_frames - 1, step=1, initial_value=0
            )
            self._gui_end_frame = self._server.gui.add_slider(
                "End Frame", min=0, max=num_frames - 1, step=1, initial_value=num_frames - 1
            )
            self._gui_background = self._server.gui.add_rgb(
                "Background", initial_value=DARK_BACKGROUND if self._context.dark_mode else LIGHT_BACKGROUND
            )
            render_button = self._server.gui.add_button("Render Scene")
            render_button.on_click(self._on_render)

            @self._gui_format.on_update
            def _on_format_changed(_) -> None:
                assert self._gui_format is not None, "GUI must be created before handling format change."
                self._config.format = self._gui_format.value

            @self._gui_view.on_update
            def _on_view_changed(_) -> None:
                assert self._gui_view is not None, "GUI must be created before handling view change."
                self._config.view = self._gui_view.value

            @self._resolution.on_update
            def _on_resolution_changed(_) -> None:
                assert self._resolution is not None, "GUI must be created before handling resolution change."
                self._config.resolution = self._resolution.value

            @self._fps.on_update
            def _on_fps_changed(_) -> None:
                assert self._fps is not None, "GUI must be created before handling FPS change."
                self._config.fps = self._fps.value

            # Keep the range valid: dragging one handle past the other pushes it along.
            @self._gui_start_frame.on_update
            def _on_start_frame_changed(_) -> None:
                assert self._gui_start_frame is not None and self._gui_end_frame is not None
                if self._gui_start_frame.value > self._gui_end_frame.value:
                    self._gui_end_frame.value = self._gui_start_frame.value

            @self._gui_end_frame.on_update
            def _on_end_frame_changed(_) -> None:
                assert self._gui_start_frame is not None and self._gui_end_frame is not None
                if self._gui_end_frame.value < self._gui_start_frame.value:
                    self._gui_start_frame.value = self._gui_end_frame.value

    def _composite_over_background(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Alpha-composite an RGBA frame over the configured background color."""
        assert self._gui_background is not None, "GUI must be created before compositing."
        if image.shape[-1] == 3:
            return image
        rgb = image[..., :3].astype(np.uint16)
        alpha = image[..., 3:4].astype(np.uint16)
        background = np.asarray(self._gui_background.value, dtype=np.uint16)
        return ((rgb * alpha + background * (255 - alpha)) // 255).astype(np.uint8)

    def _on_render(self, event: viser.GuiEvent) -> None:
        assert self._gui_format is not None, "GUI must be created before handling render."
        assert self._gui_view is not None, "GUI must be created before handling render."
        assert self._gui_start_frame is not None and self._gui_end_frame is not None
        client = event.client
        if client is None:
            return

        client.scene.reset()
        self._playback.is_rendering = True
        try:
            images = []
            scene = self._context.scene
            initial_ego_state = self._context.initial_ego_state

            width, height = RESOLUTION_MAP[self._config.resolution]

            start_frame = min(self._gui_start_frame.value, self._gui_end_frame.value)
            end_frame = max(self._gui_start_frame.value, self._gui_end_frame.value)
            for i in tqdm(range(start_frame, end_frame + 1)):
                self._playback.set_timestep(i)
                if self._gui_view.value == "BEV":
                    ego_view = get_ego_bev_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                elif self._gui_view.value == "3rd Person":
                    ego_view = get_ego_3rd_person_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                # PNG transport renders on a transparent background (jpeg hardcodes white
                # in the viser frontend); the background color is composited below.
                images.append(client.get_render(height=height, width=width, transport_format="png"))

            format = self._gui_format.value
            content: Optional[bytes] = None
            if format == "gif":
                composited = [self._composite_over_background(image) for image in images]
                buffer = io.BytesIO()
                iio.imwrite(buffer, composited, extension=".gif", loop=False)
                content = buffer.getvalue()
            elif format == "mp4":
                composited = [self._composite_over_background(image) for image in images]
                buffer = io.BytesIO()
                fps = int(self._config.fps.split()[0])
                # The pyav plugin does not infer a codec from the container format.
                iio.imwrite(buffer, composited, extension=".mp4", fps=fps, codec="libx264")
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

            assert content is not None, "Content should have been generated by this point."
            scene_name = f"{scene.log_metadata.split}_{scene.scene_uuid}"
            client.send_file_download(f"{scene_name}.{format}", content)
        finally:
            self._playback.is_rendering = False
