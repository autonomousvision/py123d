import io
import logging
import zipfile
from dataclasses import dataclass
from typing import List, Literal, Optional

import av
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import viser
from PIL import Image
from tqdm import tqdm

from py123d.visualization.viser.camera_strip_controller import CameraStripController
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.utils.view_utils import (
    FollowAnchor,
    follow_camera_pose,
    get_ego_3rd_person_view_position,
    get_ego_bev_view_position,
)

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


# Default length of the render range ahead of the current frame.
DEFAULT_RENDER_RANGE_FRAMES = 100

# libx264 constant rate factors for mp4 exports, shown as user-facing labels. The
# imageio pyav wrapper offers no quality control and falls back to the x264 default
# (23), which compresses point clouds and camera imagery visibly.
MP4_QUALITY_MAP = {
    "very high (crf 14)": 14,
    "high (crf 16)": 16,
    "medium (crf 20)": 20,
    "low (crf 23)": 23,
}


@dataclass
class RenderConfig:
    format: Literal["gif", "mp4", "png"] = "mp4"
    view: Literal["Follow", "3rd Person", "BEV", "Manual"] = "Follow"
    resolution: Literal["480p", "720p", "1080p", "1440p", "4K"] = "1080p"
    fps: Literal["5 fps", "10 fps", "15 fps", "20 fps", "30 fps"] = "20 fps"
    quality: Literal["very high (crf 14)", "high (crf 16)", "medium (crf 20)", "low (crf 23)"] = "high (crf 16)"


def _encode_mp4(frames: List[npt.NDArray[np.uint8]], fps: int, crf: int) -> bytes:
    """Encode RGB frames as h264 mp4 with an explicit quality setting.

    Encodes via pyav directly because the imageio pyav plugin exposes no encoder
    options, leaving libx264 at its default (visibly lossy) rate factor.
    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image[..., :3]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return buffer.getvalue()


class RenderController:
    """Manages the render-to-file workflow (gif, mp4, png)."""

    def __init__(
        self,
        server: viser.ViserServer,
        config: RenderConfig,
        context: ElementContext,
        playback_controller: PlaybackController,
        camera_strip: Optional[CameraStripController] = None,
    ) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._playback = playback_controller
        self._camera_strip = camera_strip
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
                "View", ["Follow", "3rd Person", "BEV", "Manual"], initial_value=self._config.view
            )
            self._resolution = self._server.gui.add_dropdown(
                "Resolution", ["480p", "720p", "1080p", "1440p", "4K"], initial_value=self._config.resolution
            )
            self._fps = self._server.gui.add_dropdown(
                "FPS", ["5 fps", "10 fps", "15 fps", "20 fps", "30 fps"], initial_value=self._config.fps
            )
            self._gui_quality = self._server.gui.add_dropdown(
                "Quality", list(MP4_QUALITY_MAP), initial_value=self._config.quality
            )
            num_frames = self._context.num_frames
            self._gui_start_frame = self._server.gui.add_slider(
                "Start Frame", min=0, max=num_frames - 1, step=1, initial_value=0
            )
            self._gui_end_frame = self._server.gui.add_slider(
                "End Frame",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=min(DEFAULT_RENDER_RANGE_FRAMES, num_frames - 1),
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

            @self._gui_quality.on_update
            def _on_quality_changed(_) -> None:
                assert self._gui_quality is not None, "GUI must be created before handling quality change."
                self._config.quality = self._gui_quality.value

            # Keep the range valid: dragging one handle past the other pushes it along.
            @self._gui_start_frame.on_update
            def _on_start_frame_changed(_) -> None:
                assert self._gui_start_frame is not None and self._gui_end_frame is not None
                new_end = max(self._gui_end_frame.value, self._gui_start_frame.value)
                if new_end != self._gui_end_frame.value:
                    self._gui_end_frame.value = new_end

            @self._gui_end_frame.on_update
            def _on_end_frame_changed(_) -> None:
                assert self._gui_start_frame is not None and self._gui_end_frame is not None
                new_start = min(self._gui_start_frame.value, self._gui_end_frame.value)
                if new_start != self._gui_start_frame.value:
                    self._gui_start_frame.value = new_start

    def set_default_frame_range(self, iteration: int) -> None:
        """Track playback: the render range defaults to the current frame plus the
        next ``DEFAULT_RENDER_RANGE_FRAMES`` frames (clamped to the scene end), so a
        render started after scrubbing/playing captures what is currently on screen."""
        if self._gui_start_frame is None or self._gui_end_frame is None:
            return
        if self._playback.is_rendering:
            return
        last_frame = self._context.num_frames - 1
        start = min(iteration, last_frame)
        end = min(iteration + DEFAULT_RENDER_RANGE_FRAMES, last_frame)
        if self._gui_start_frame.value != start:
            self._gui_start_frame.value = start
        if self._gui_end_frame.value != end:
            self._gui_end_frame.value = end

    def _ego_position(self, iteration: int) -> npt.NDArray[np.float64]:
        """Scene-centered ego position at the given iteration."""
        state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert state is not None, f"Ego state must be available at iteration {iteration}."
        return state.center_se3.point_3d.array.astype(np.float64) - self._context.scene_center_array.astype(np.float64)

    def _ego_yaw(self, iteration: int) -> float:
        """Planar ego heading at the given iteration."""
        state = self._context.scene.get_ego_state_se3_at_iteration(iteration)
        assert state is not None, f"Ego state must be available at iteration {iteration}."
        return float(state.center_se3.yaw)

    def _composite_camera_strip(self, frame: npt.NDArray[np.uint8], iteration: int) -> npt.NDArray[np.uint8]:
        """Rasterize the enabled camera strip into the top of a rendered frame.

        Mirrors the HTML overlay layout: each image one third of the frame width,
        row centered horizontally. No-op when the strip is disabled."""
        if self._camera_strip is None:
            return frame
        labeled_images = self._camera_strip.get_enabled_images(iteration)
        if len(labeled_images) == 0:
            return frame
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        slot_width = frame_width // 3
        resized = []
        for _, image in labeled_images:
            scale = min(slot_width / image.shape[1], frame_height / image.shape[0])
            new_w = max(1, int(round(image.shape[1] * scale)))
            new_h = max(1, int(round(image.shape[0] * scale)))
            resized.append(np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)))
        total_width = sum(img.shape[1] for img in resized)
        x = max(0, (frame_width - total_width) // 2)
        for image in resized:
            h, w = image.shape[:2]
            h = min(h, frame_height)
            w = min(w, frame_width - x)
            if w <= 0:
                break
            frame[:h, x : x + w, :3] = image[:h, :w, :3]
            if frame.shape[-1] == 4:
                frame[:h, x : x + w, 3] = 255
            x += w
        return frame

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

            # Follow view: the user's current camera, rigidly attached to the vehicle's
            # planar body frame via the same follow_camera_pose used by the playback
            # ego-follow -- what-you-see-is-what-you-get for the pose the viewer shows
            # when the render starts, rotating with the vehicle.
            follow_anchor: Optional[FollowAnchor] = None
            if self._gui_view.value == "Follow":
                try:
                    current_iteration = self._playback.current_iteration
                    follow_anchor = FollowAnchor(
                        ego_position=self._ego_position(current_iteration),
                        ego_yaw=self._ego_yaw(current_iteration),
                        camera_position=np.asarray(client.camera.position, dtype=np.float64),
                        camera_wxyz=np.asarray(client.camera.wxyz, dtype=np.float64),
                    )
                except AssertionError:
                    # No camera state received yet; fall back to a static camera.
                    follow_anchor = None

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
                elif self._gui_view.value == "Follow" and follow_anchor is not None:
                    position, wxyz = follow_camera_pose(follow_anchor, self._ego_position(i), self._ego_yaw(i))
                    client.camera.position = position
                    client.camera.wxyz = wxyz
                # PNG transport renders on a transparent background (jpeg hardcodes white
                # in the viser frontend); the background color is composited below.
                frame = client.get_render(height=height, width=width, transport_format="png")
                frame = self._composite_camera_strip(frame, i)
                images.append(frame)

            format = self._gui_format.value
            content: Optional[bytes] = None
            if format == "gif":
                composited = [self._composite_over_background(image) for image in images]
                buffer = io.BytesIO()
                iio.imwrite(buffer, composited, extension=".gif", loop=False)
                content = buffer.getvalue()
            elif format == "mp4":
                composited = [self._composite_over_background(image) for image in images]
                fps = int(self._config.fps.split()[0])
                content = _encode_mp4(composited, fps=fps, crf=MP4_QUALITY_MAP[self._config.quality])
            elif format == "png":
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for idx, img in enumerate(images):
                        name = f"frame_{idx:05d}.png"
                        if isinstance(img, (bytes, bytearray)):
                            zf.writestr(name, img)
                        else:
                            # Frames are RGBA on a transparent background; honor the
                            # Background control like the gif/mp4 paths do.
                            img_bytes = io.BytesIO()
                            iio.imwrite(img_bytes, self._composite_over_background(img), extension=".png")
                            zf.writestr(name, img_bytes.getvalue())
                content = zip_buf.getvalue()
                format = "zip"

            assert content is not None, "Content should have been generated by this point."
            scene_name = f"{scene.log_metadata.split}_{scene.scene_uuid}"
            client.send_file_download(f"{scene_name}.{format}", content)
        finally:
            self._playback.is_rendering = False
