import logging
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.camera_gui_controller import CameraGuiController
from py123d.visualization.viser.camera_strip_controller import CameraStripController
from py123d.visualization.viser.element_manager import ElementManager
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.elements.box_detections_se3_element import BoxDetectionsSE3Element
from py123d.visualization.viser.elements.camera_frustum_element import CameraFrustumElement
from py123d.visualization.viser.elements.ego_state_se3_element import EgoElement
from py123d.visualization.viser.elements.lidar_element import LidarElement
from py123d.visualization.viser.elements.map_element import MapElement
from py123d.visualization.viser.elements.radar_element import RadarElement
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.render_controller import RenderController
from py123d.visualization.viser.viser_config import ViserConfig

logger = logging.getLogger(__name__)

HDRI: Literal[
    "apartment",
    "city",
    "dawn",
    "forest",
    "lobby",
    "night",
    "park",
    "studio",
    "sunset",
    "warehouse",
] = "warehouse"


# Global CSS injected once per scene via an (invisible) HTML GUI element. Packs
# consecutive checkbox rows into two columns to keep the panel compact: the outer
# per-input wrapper of a checkbox becomes an inline 50%-width element, so adjacent
# checkboxes flow next to each other while any other component breaks the row.
#
# The wrapper is identified by its exact internal chain (wrapper > flex > first
# label box > <p> > <label>) AND containing a checkbox; ancestors like the folder
# content box fail the depth-exact label chain, and sliders/dropdowns fail the
# checkbox condition. Verified against the DOM structure of viser 1.0.24
# (Generated.tsx / common.tsx / Checkbox.tsx, Mantine 8).
_COMPACT_GUI_CSS = """<style>
div:has(> div > div:first-child > p > label):has(.mantine-Checkbox-root) {
  display: inline-flex;
  width: 50%;
  box-sizing: border-box;
  vertical-align: top;
  min-width: 0;
}
/* Checkbox before its label: reorder the flex children (label box first in the DOM)
   and stop the checkbox container from growing, so the pair packs to the left. */
div:has(> div > div:first-child > p > label):has(.mantine-Checkbox-root) > div {
  justify-content: flex-start;
}
div:has(> div > div:first-child > p > label):has(.mantine-Checkbox-root) > div > div:first-child {
  order: 2;
  width: auto !important;
  padding-right: 0 !important;
}
div:has(> div > div:first-child > p > label):has(.mantine-Checkbox-root) > div > div:last-child {
  order: 1;
  flex-grow: 0 !important;
  margin-right: 0.45em;
}
/* The camera strip renders at z-index -1 inside the control panel's stacking
   context (below all controls). The panel root's own background paints below
   negative-z descendants, so give the panel content wrapper an opaque inherited
   background: in-flow wrapper backgrounds paint above negative-z elements and
   block the strip from shining through gaps between folder cards. */
.mantine-Paper-root > div:has(.py123d-camera-strip) {
  background: inherit;
  border-radius: inherit;
}
</style>"""


# Cold scene loading is dominated by parsing the per-sensor arrow tables, whose
# record-batch count scales with the iteration count. Calibrated on a 67886-iteration
# log loading in ~20 s with the parallel fetch path.
_COLD_LOAD_SECONDS_PER_ITERATION = 3.0e-4
# No overlay when the expected load time is below this.
_LOADING_OVERLAY_MIN_SECONDS = 1.0


def _loading_overlay_html(scene_name: str, estimate_s: float) -> str:
    rounding = 1.0 if estimate_s < 10.0 else 5.0
    estimate = max(rounding, round(estimate_s / rounding) * rounding)
    return (
        '<div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;'
        'z-index:2000;pointer-events:none;">'
        '<div style="background:rgba(20,21,23,0.92);color:#ffffff;font-size:2.0em;font-weight:600;'
        'padding:0.8em 1.4em;border-radius:0.4em;text-align:center;">'
        f"Loading {scene_name}&hellip;<br/>"
        f'<span style="font-size:0.6em;font-weight:400;">(expected ~{estimate:.0f} s)</span>'
        "</div></div>"
    )


def _build_titlebar() -> TitlebarConfig:
    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://kesai.eu/py123d",
        ),
        TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/kesai-labs/py123d",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://kesai.eu/py123d",
        ),
    )
    image = TitlebarImage(
        image_url_light="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg",
        image_url_dark="https://kesai.eu/py123d/_static/123D_logo_transparent_white.svg",
        image_alt="123D",
        href="https://kesai.eu/py123d/",
    )
    return TitlebarConfig(buttons=buttons, image=image)


def _build_viser_server(config: ViserConfig) -> Tuple[viser.ViserServer, TitlebarConfig]:
    server = viser.ViserServer(
        host=config.server.host,
        port=config.server.port,
        label=config.server.label,
        verbose=config.server.verbose,
    )

    titlebar_theme = _build_titlebar()

    server.gui.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout=config.theme.control_layout,
        control_width=config.theme.control_width,
        dark_mode=config.theme.dark_mode,
        show_logo=config.theme.show_logo,
        show_share_button=config.theme.show_share_button,
        brand_color=config.theme.brand_color,
    )

    server.scene.configure_environment_map(
        hdri=HDRI,
        environment_intensity=0.75,  # down from default 1.0
    )
    return server, titlebar_theme


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
        self._server, self._titlebar = _build_viser_server(self._config)
        self._dark_mode = self._config.theme.dark_mode
        self._environment_intensity = 0.25
        self._loaded_scene_uuids: set = set()

        # Ego-follow state. Follow is absolute and idempotent: per client we keep the
        # camera offset relative to the ego and set position = ego + offset every
        # timestep. Camera echoes from the browser that do not match a recently sent
        # target are user adjustments (orbit/pan/zoom) and re-base the offset.
        self._follow_enabled: bool = False
        self._follow_scene_center: Optional[np.ndarray] = None
        self._follow_current_ego: Optional[np.ndarray] = None
        self._follow_offsets: Dict[int, np.ndarray] = {}
        self._follow_recent_targets: Dict[int, Deque[np.ndarray]] = {}

        @self._server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            @client.camera.on_update
            def _(_camera) -> None:
                self._on_client_camera_update(client)

        self._run_scene(self._scenes[self._scene_index % len(self._scenes)])

    def _run_scene(self, scene: SceneAPI) -> None:
        """Set up and run the viewer for a single scene. Blocks until scene switch."""
        # Loading overlay: shown while the cold scene setup runs, skipped for scenes
        # expected to load quickly (small logs, warm revisits). The message is flushed
        # to clients before the blocking load starts.
        loading_overlay = None
        if scene.scene_uuid not in self._loaded_scene_uuids:
            estimate_s = scene.number_of_iterations * _COLD_LOAD_SECONDS_PER_ITERATION
            if estimate_s > _LOADING_OVERLAY_MIN_SECONDS:
                loading_overlay = self._server.gui.add_html(
                    _loading_overlay_html(scene.log_metadata.log_name, estimate_s)
                )
                self._server.flush()

        # The overlay must come down even when scene setup fails, otherwise it blocks
        # the view permanently.
        try:
            context = ElementContext.from_scene(scene, dark_mode=self._dark_mode)

            # Build elements based on available data
            self._element_manager = self._build_elements(context)

            # Build controllers
            playback = PlaybackController(
                self._server,
                self._config.playback,
                context,
                on_dark_mode_changed=self._on_dark_mode_changed,
                scene_index=self._scene_index % len(self._scenes),
                num_scenes=len(self._scenes),
            )
            # Build camera GUI controller
            self._camera_gui = CameraGuiController(self._server, self._config.camera_gui, context)

            # Build camera strip overlay (full viewport width).
            self._camera_strip = CameraStripController(self._server, self._config.camera_strip, context)

            # The render controller rasterizes the enabled camera strip into exports.
            render = RenderController(
                self._server, self._config.render, context, playback, camera_strip=self._camera_strip
            )

            # Create GUI in order: Playback -> Modality Tabs -> Camera Image -> Camera Strip -> Render
            self._server.gui.add_html(_COMPACT_GUI_CSS)
            playback.create_gui(scene)
            self._element_manager.create_all_gui(self._server)
            self._camera_gui.create_gui()
            self._camera_strip.create_gui()
            render.create_gui()

            # Re-apply persisted environment intensity (scene.reset() clears it)
            self._server.scene.configure_environment_map(
                hdri=HDRI,
                environment_intensity=self._environment_intensity,
            )

            # Wire iteration callback
            self._follow_scene_center = context.scene_center_array.astype(np.float64)
            self._follow_current_ego = None
            self._follow_offsets.clear()
            self._follow_recent_targets.clear()

            def _on_iteration_changed(iteration: int) -> None:
                # Follow is suspended while rendering: the render controller drives the
                # camera itself, and its camera echoes must not re-base the follow offset
                # (which would leave the interactive camera on the render path afterwards).
                # The disabled path clears the follow state, so follow re-engages from the
                # current camera on the first timestep after the render.
                self._apply_ego_follow(
                    scene, iteration, follow_enabled=playback.follow_ego and not playback.is_rendering
                )
                self._element_manager.update_all(iteration)
                self._camera_gui.update(iteration)
                self._camera_strip.update(iteration)
                render.set_default_frame_range(iteration)

            playback.set_on_iteration_changed(_on_iteration_changed)

            # Initial render at frame 0
            _on_iteration_changed(0)

            self._loaded_scene_uuids.add(scene.scene_uuid)
        finally:
            if loading_overlay is not None:
                loading_overlay.remove()

        # Blocking playback loop -- returns on Next Scene
        playback.run_loop()

        # Cleanup and advance to next scene
        self._camera_gui.remove()
        self._camera_strip.remove()
        self._element_manager.remove_all()
        self._server.flush()
        self._server.gui.reset()
        self._server.scene.reset()
        requested_scene_index = playback.requested_scene_index
        if requested_scene_index is not None:
            self._scene_index = requested_scene_index % len(self._scenes)
        else:
            self._scene_index = (self._scene_index + 1) % len(self._scenes)
        self._run_scene(self._scenes[self._scene_index])

    def _apply_ego_follow(self, scene: SceneAPI, iteration: int, follow_enabled: bool) -> None:
        """Keep each client camera at a fixed offset to the ego vehicle.

        Absolute placement (position = ego + per-client offset) instead of
        incremental deltas: incremental shifts race the throttled camera echoes
        from the browser, and every shift applied to a stale state is lost
        permanently, drifting the camera off the vehicle. Absolute targets are
        idempotent, so lost or reordered messages cause no accumulating error.
        Zoom, pan offset, and orientation are preserved (the viser position
        setter moves the look-at point along).
        """
        self._follow_enabled = follow_enabled
        if not follow_enabled:
            self._follow_offsets.clear()
            self._follow_recent_targets.clear()
            return
        state = scene.get_ego_state_se3_at_iteration(iteration)
        if state is None or self._follow_scene_center is None:
            return
        ego = state.center_se3.point_3d.array.astype(np.float64) - self._follow_scene_center
        self._follow_current_ego = ego

        for client in self._server.get_clients().values():
            try:
                camera_position = np.asarray(client.camera.position, dtype=np.float64)
            except AssertionError:
                # Camera state has not been received from this client yet.
                continue
            offset = self._follow_offsets.get(client.client_id)
            if offset is None:
                # Follow engages from wherever the camera currently is.
                offset = camera_position - ego
                self._follow_offsets[client.client_id] = offset
            target = ego + offset
            client.camera.position = target
            # Long history (~6 s at 10 Hz): a delayed echo of our own target must
            # never be mistaken for a user camera move, which would re-base the
            # offset to a lagged position and jerk the viewport.
            targets = self._follow_recent_targets.setdefault(client.client_id, deque(maxlen=60))
            targets.append(target)

    def _on_client_camera_update(self, client: viser.ClientHandle) -> None:
        """Re-base the follow offset when the browser reports a user camera move.

        Echoes of our own follow targets arrive delayed; anything matching a
        recently sent target is ignored, everything else is a manual adjustment
        and becomes the new camera-to-ego offset.
        """
        if not self._follow_enabled or self._follow_current_ego is None:
            return
        try:
            camera_position = np.asarray(client.camera.position, dtype=np.float64)
        except AssertionError:
            return
        targets = self._follow_recent_targets.get(client.client_id)
        if targets is not None:
            for target in list(targets):
                if float(np.linalg.norm(camera_position - target)) < 0.5:
                    return
        self._follow_offsets[client.client_id] = camera_position - self._follow_current_ego

    def _on_dark_mode_changed(self, dark_mode: bool) -> None:
        """Handle dark mode toggle from playback controller."""
        theme = self._config.theme
        self._dark_mode = dark_mode
        self._server.gui.configure_theme(
            titlebar_content=self._titlebar,
            control_layout=theme.control_layout,
            control_width=theme.control_width,
            dark_mode=dark_mode,
            show_logo=theme.show_logo,
            show_share_button=theme.show_share_button,
            brand_color=theme.brand_color,
        )
        self._element_manager.notify_dark_mode_changed(dark_mode)

    def _build_elements(self, context: ElementContext) -> ElementManager:
        """Conditionally register elements based on what the scene supports."""
        manager = ElementManager()
        scene = context.scene

        if len(scene.get_lidar_metadatas()) > 0:
            manager.register(LidarElement(context, self._config.lidar))

        if len(scene.get_radar_metadatas()) > 0:
            manager.register(RadarElement(context, self._config.radar))

        if scene.get_box_detections_se3_metadata() is not None:
            manager.register(BoxDetectionsSE3Element(context, self._config.detection))

        if len(scene.get_camera_metadatas()) > 0:
            manager.register(CameraFrustumElement(context, self._config.camera_frustum))

        if scene.get_ego_state_se3_metadata() is not None:
            manager.register(EgoElement(context, self._config.ego))

        if scene.get_map_api() is not None:
            manager.register(MapElement(context, self._config.map))

        return manager
