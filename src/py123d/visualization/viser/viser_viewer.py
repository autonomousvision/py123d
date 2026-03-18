import logging
from typing import List

import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.element_manager import ElementManager
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.elements.camera_frustum_element import CameraFrustumElement
from py123d.visualization.viser.elements.camera_gui_element import CameraGuiElement
from py123d.visualization.viser.elements.detection_element import DetectionElement
from py123d.visualization.viser.elements.lidar_element import LidarElement
from py123d.visualization.viser.elements.map_element import MapElement
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.render_controller import RenderController
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)


def _build_viser_server(config: ViserConfig) -> viser.ViserServer:
    server = viser.ViserServer(
        host=config.server.host,
        port=config.server.port,
        label=config.server.label,
        verbose=config.server.verbose,
    )

    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://autonomousvision.github.io/py123d",
        ),
        TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/autonomousvision/py123d",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://autonomousvision.github.io/py123d",
        ),
    )
    image = TitlebarImage(
        image_url_light="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_black.svg",
        image_url_dark="https://autonomousvision.github.io/py123d/_static/123D_logo_transparent_white.svg",
        image_alt="123D",
        href="https://autonomousvision.github.io/py123d/",
    )
    titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

    server.gui.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout=config.theme.control_layout,
        control_width=config.theme.control_width,
        dark_mode=config.theme.dark_mode,
        show_logo=config.theme.show_logo,
        show_share_button=config.theme.show_share_button,
        brand_color=config.theme.brand_color,
    )
    return server


class ViserViewer:
    """Orchestrates the viser 3D viewer: wires elements, playback, and rendering together."""

    def __init__(
        self,
        scenes: List[SceneAPI],
        viser_config: ViserConfig = ViserConfig(),
        scene_index: int = 0,
    ) -> None:
        if len(scenes) == 0:
            raise ValueError("At least one scene must be provided.")

        self._scenes = scenes
        self._config = viser_config
        self._scene_index = scene_index
        self._server = _build_viser_server(self._config)
        self._run_scene(self._scenes[self._scene_index % len(self._scenes)])

    def _run_scene(self, scene: SceneAPI) -> None:
        """Set up and run the viewer for a single scene. Blocks until scene switch."""
        context = ElementContext.from_scene(scene)

        # Build elements based on available data
        element_manager = self._build_elements(context)

        # Build controllers
        playback = PlaybackController(self._server, self._config.playback, context)
        render = RenderController(self._server, context, playback)

        # Create GUI in order: Playback -> Modality Tabs -> Render
        playback.create_gui(scene)
        element_manager.create_all_gui(self._server)
        render.create_gui()

        # Wire iteration callback
        playback.set_on_iteration_changed(element_manager.update_all)

        # Initial render at frame 0
        element_manager.update_all(0)

        # Blocking playback loop -- returns on Next Scene
        playback.run_loop()

        # Cleanup and advance to next scene
        element_manager.remove_all()
        self._server.flush()
        self._server.gui.reset()
        self._server.scene.reset()
        self._scene_index = (self._scene_index + 1) % len(self._scenes)
        self._run_scene(self._scenes[self._scene_index])

    def _build_elements(self, context: ElementContext) -> ElementManager:
        """Conditionally register elements based on what the scene supports."""
        manager = ElementManager()
        scene = context.scene

        if len(scene.available_lidar_ids) > 0:
            manager.register(LidarElement(context, self._config.lidar))

        # Single camera frustum element handles both pinhole and fisheye
        all_camera_types = list(self._config.camera_frustum.pinhole_types) + list(
            self._config.camera_frustum.fisheye_types
        )
        available_camera_types = set(all_camera_types) & set(scene.available_camera_ids)
        if len(available_camera_types) > 0:
            manager.register(CameraFrustumElement(context, self._config.camera_frustum))

        if len(scene.available_camera_ids) > 0:
            manager.register(CameraGuiElement(context, self._config.camera_gui))

        manager.register(DetectionElement(context, self._config.detection))

        if scene.get_map_api() is not None:
            manager.register(MapElement(context, self._config.map))

        return manager
