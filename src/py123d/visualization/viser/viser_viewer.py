import io
import logging
import time
from typing import Dict, List, Optional

import imageio.v3 as iio
import viser
from tqdm import tqdm
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from py123d.datatypes.maps.map_datatypes import MapLayer
from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraType
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.visualization.viser.elements import (
    add_box_detections_to_viser_server,
    add_camera_frustums_to_viser_server,
    add_camera_gui_to_viser_server,
    add_lidar_pc_to_viser_server,
    add_map_to_viser_server,
)
from py123d.visualization.viser.elements.render_elements import (
    get_ego_3rd_person_view_position,
    get_ego_bev_view_position,
)
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)


def _build_viser_server(viser_config: ViserConfig) -> viser.ViserServer:
    server = viser.ViserServer(
        host=viser_config.server_host,
        port=viser_config.server_port,
        label=viser_config.server_label,
        verbose=viser_config.server_verbose,
    )

    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://danieldauner.github.io/123d",
        ),
        TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/DanielDauner/123d",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://danieldauner.github.io/123d",
        ),
    )
    image = TitlebarImage(
        image_url_light="https://danieldauner.github.io/123d/_static/logo_black.png",
        image_url_dark="https://danieldauner.github.io/123d/_static/logo_white.png",
        image_alt="123D",
        href="https://danieldauner.github.io/123d/",
    )
    titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

    server.gui.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout=viser_config.theme_control_layout,
        control_width=viser_config.theme_control_width,
        dark_mode=viser_config.theme_dark_mode,
        show_logo=viser_config.theme_show_logo,
        show_share_button=viser_config.theme_show_share_button,
        brand_color=viser_config.theme_brand_color,
    )
    return server


class ViserViewer:
    def __init__(
        self,
        scenes: List[AbstractScene],
        viser_config: ViserConfig = ViserConfig(),
        scene_index: int = 0,
    ) -> None:
        assert len(scenes) > 0, "At least one scene must be provided."

        self._scenes = scenes
        self._viser_config = viser_config
        self._scene_index = scene_index

        self._viser_server = _build_viser_server(self._viser_config)
        self.set_scene(self._scenes[self._scene_index % len(self._scenes)])

    def next(self) -> None:
        self._viser_server.flush()
        self._viser_server.gui.reset()
        self._viser_server.scene.reset()
        self._scene_index = (self._scene_index + 1) % len(self._scenes)
        self.set_scene(self._scenes[self._scene_index])

    def set_scene(self, scene: AbstractScene) -> None:
        num_frames = scene.number_of_iterations
        initial_ego_state: EgoStateSE3 = scene.get_ego_state_at_iteration(0)
        server_playing = True
        server_rendering = False

        with self._viser_server.gui.add_folder("Playback"):
            gui_timestep = self._viser_server.gui.add_slider(
                "Timestep",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_next_frame = self._viser_server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = self._viser_server.gui.add_button("Prev Frame", disabled=True)
            gui_next_scene = self._viser_server.gui.add_button("Next Scene", disabled=False)
            gui_playing = self._viser_server.gui.add_checkbox("Playing", True)
            gui_framerate = self._viser_server.gui.add_slider("FPS", min=1, max=100, step=1, initial_value=10)
            gui_framerate_options = self._viser_server.gui.add_button_group(
                "FPS options", ("10", "25", "50", "75", "100")
            )

        with self._viser_server.gui.add_folder("Render", expand_by_default=False):
            render_format = self._viser_server.gui.add_dropdown("Format", ["gif", "mp4"], initial_value="mp4")
            render_view = self._viser_server.gui.add_dropdown(
                "View", ["3rd Person", "BEV", "Manual"], initial_value="3rd Person"
            )
            button = self._viser_server.gui.add_button("Render Scene")

        # Frame step buttons.
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        @gui_next_scene.on_click
        def _(_) -> None:
            nonlocal server_playing
            server_playing = False

        # Disable frame controls when we're playing.
        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        # Set the framerate when we click one of the options.
        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        # Toggle frame visibility when the timestep slider changes.
        @gui_timestep.on_update
        def _(_) -> None:
            start = time.perf_counter()
            add_box_detections_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
            )
            add_camera_frustums_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                camera_frustum_handles,
            )
            add_camera_gui_to_viser_server(
                scene,
                gui_timestep.value,
                self._viser_server,
                self._viser_config,
                camera_gui_handles,
            )
            add_lidar_pc_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                lidar_pc_handle,
            )
            add_map_to_viser_server(
                scene,
                gui_timestep.value,
                initial_ego_state,
                self._viser_server,
                self._viser_config,
                map_handles,
            )
            rendering_time = time.perf_counter() - start
            sleep_time = 1.0 / gui_framerate.value - rendering_time
            if sleep_time > 0 and not server_rendering:
                time.sleep(max(sleep_time, 0.0))

        @button.on_click
        def _(event: viser.GuiEvent) -> None:
            nonlocal server_rendering
            client = event.client
            assert client is not None

            client.scene.reset()

            server_rendering = True
            images = []

            for i in tqdm(range(scene.number_of_iterations)):
                gui_timestep.value = i
                if render_view.value == "BEV":
                    ego_view = get_ego_bev_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                elif render_view.value == "3rd Person":
                    ego_view = get_ego_3rd_person_view_position(scene, i, initial_ego_state)
                    client.camera.position = ego_view.point_3d.array
                    client.camera.wxyz = ego_view.quaternion.array
                images.append(client.get_render(height=1080, width=1920))
            format = render_format.value
            buffer = io.BytesIO()
            if format == "gif":
                iio.imwrite(buffer, images, extension=".gif", loop=False)
            elif format == "mp4":
                iio.imwrite(buffer, images, extension=".mp4", fps=20)
            content = buffer.getvalue()
            scene_name = f"{scene.log_metadata.split}_{scene.uuid}"
            client.send_file_download(f"{scene_name}.{format}", content, save_immediately=True)
            server_rendering = False

        camera_frustum_handles: Dict[PinholeCameraType, viser.CameraFrustumHandle] = {}
        camera_gui_handles: Dict[PinholeCameraType, viser.GuiImageHandle] = {}
        lidar_pc_handle: Optional[viser.PointCloudHandle] = None
        map_handles: Dict[MapLayer, viser.MeshHandle] = {}

        add_box_detections_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
        )
        add_camera_frustums_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            camera_frustum_handles,
        )
        add_camera_gui_to_viser_server(
            scene,
            gui_timestep.value,
            self._viser_server,
            self._viser_config,
            camera_gui_handles,
        )
        add_lidar_pc_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            lidar_pc_handle,
        )
        add_map_to_viser_server(
            scene,
            gui_timestep.value,
            initial_ego_state,
            self._viser_server,
            self._viser_config,
            map_handles,
        )

        # Playback update loop.
        while server_playing:

            if gui_playing.value and not server_rendering:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            else:
                time.sleep(0.1)

        self._viser_server.flush()
        self.next()
