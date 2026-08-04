import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.elements.base_element import ElementContext

logger = logging.getLogger(__name__)

# The viser number fields next to sliders emit an update per keystroke. User-originated
# updates are therefore debounced: only the value that is stable for this long is
# applied. Programmatic updates (playback loop, render controller) are never debounced.
_TIMESTEP_DEBOUNCE_S = 0.3
_SCENE_DEBOUNCE_S = 0.5


@dataclass
class PlaybackConfig:
    is_playing: bool = False
    speed: float = 1.0
    atomic: bool = False
    dark_mode: bool = True  # kept in sync with the theme; initialization uses ThemeConfig.dark_mode
    follow_ego: bool = False


class PlaybackController:
    """Manages playback state, timestep navigation, and the blocking playback loop."""

    def __init__(
        self,
        server: viser.ViserServer,
        config: PlaybackConfig,
        context: ElementContext,
        on_dark_mode_changed: Optional[Callable[[bool], None]] = None,
        scene_index: int = 0,
        num_scenes: int = 1,
    ) -> None:
        self._server = server
        self._config = config
        self._context = context
        self._scene_index = scene_index
        self._num_scenes = num_scenes
        self._should_stop: bool = False
        self._rendering: bool = False
        self._requested_scene_index: Optional[int] = None
        self._on_iteration_changed: Optional[Callable[[int], None]] = None
        self._on_dark_mode_changed = on_dark_mode_changed
        self._iteration_duration_s: float = 0.1
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_lock = threading.Lock()

        # GUI handles (created in create_gui)
        self._gui_timestep: Optional[viser.GuiSliderHandle] = None
        self._gui_playing: Optional[viser.GuiCheckboxHandle] = None
        self._gui_speed: Optional[viser.GuiSliderHandle] = None
        self._gui_atomic: Optional[viser.GuiCheckboxHandle] = None
        self._gui_follow_ego: Optional[viser.GuiCheckboxHandle] = None

    @property
    def current_iteration(self) -> int:
        """Current timestep value."""
        return self._gui_timestep.value if self._gui_timestep is not None else 0

    @property
    def requested_scene_index(self) -> Optional[int]:
        """Scene index requested via the scene slider or prev/next buttons, if any."""
        return self._requested_scene_index

    @property
    def follow_ego(self) -> bool:
        """Whether the camera should follow the ego vehicle across timestep changes."""
        return self._gui_follow_ego.value if self._gui_follow_ego is not None else self._config.follow_ego

    @property
    def is_rendering(self) -> bool:
        return self._rendering

    @is_rendering.setter
    def is_rendering(self, value: bool) -> None:
        self._rendering = value

    def set_on_iteration_changed(self, callback: Callable[[int], None]) -> None:
        """Set the callback invoked when the timestep changes."""
        self._on_iteration_changed = callback

    def set_timestep(self, value: int) -> None:
        """Programmatically set the timestep (used by render controller)."""
        self._gui_timestep.value = value

    def create_gui(self, scene: SceneAPI) -> None:
        """Create the Playback folder with all controls.

        Row order: scene name, scene slider, prev/next scene, timestep slider,
        prev/next frame, followed by the playback settings.
        """
        num_frames = self._context.num_frames
        controls_disabled = self._config.is_playing
        single_scene = self._num_scenes <= 1
        self._iteration_duration_s = scene.scene_metadata.iteration_duration_s

        with self._server.gui.add_folder("Playback"):
            self._server.gui.add_markdown(f"**{scene.log_metadata.log_name}**")
            gui_scene: Optional[viser.GuiSliderHandle] = None
            if not single_scene:
                gui_scene = self._server.gui.add_slider(
                    "Scene", min=0, max=self._num_scenes - 1, step=1, initial_value=self._scene_index
                )
            gui_scene_nav = self._server.gui.add_button_group("", ("Prev Scene", "Next Scene"), disabled=single_scene)
            self._gui_timestep = self._server.gui.add_slider(
                "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=controls_disabled
            )
            gui_frame_nav = self._server.gui.add_button_group(
                "", ("Prev Frame", "Next Frame"), disabled=controls_disabled
            )
            # Checkbox pairs are placed adjacently so the two-column GUI CSS packs each
            # pair into one row: (Playing, Follow Ego), (Atomic Updates, Dark Mode).
            self._gui_playing = self._server.gui.add_checkbox("Playing", self._config.is_playing)
            self._gui_follow_ego = self._server.gui.add_checkbox("Follow Ego", self._config.follow_ego)
            self._gui_speed = self._server.gui.add_slider(
                "Playback speed", min=0.1, max=10.0, step=0.1, initial_value=self._config.speed
            )
            gui_speed_options = self._server.gui.add_button_group("Options.", ("0.5", "1.0", "2.0", "5.0", "10.0"))
            self._gui_atomic = self._server.gui.add_checkbox("Atomic Updates", self._config.atomic)
            # Initialize from the live theme state (context.dark_mode) so the checkbox is
            # consistent with the actual viewer theme; ThemeConfig.dark_mode is the source
            # of truth, PlaybackConfig.dark_mode is not used for initialization.
            gui_dark_mode = self._server.gui.add_checkbox("Dark Mode", initial_value=self._context.dark_mode)

            @self._gui_atomic.on_update
            def _on_atomic_changed(_) -> None:
                self._config.atomic = self._gui_atomic.value

            @self._gui_follow_ego.on_update
            def _on_follow_ego_changed(_) -> None:
                self._config.follow_ego = self._gui_follow_ego.value

            @self._gui_speed.on_update
            def _on_speed_changed(_) -> None:
                self._config.speed = self._gui_speed.value

            @gui_dark_mode.on_update
            def _on_dark_mode_changed(_) -> None:
                self._config.dark_mode = gui_dark_mode.value
                if self._on_dark_mode_changed is not None:
                    self._on_dark_mode_changed(gui_dark_mode.value)

            # Timestep change -> update all elements. Updates from the playback loop and
            # render controller (event.client is None) apply immediately with frame
            # pacing; user edits (slider drag, number field) are debounced.
            @self._gui_timestep.on_update
            def _on_timestep_changed(event: viser.GuiEvent) -> None:
                if event.client is None:
                    self._apply_timestep(pace=True)
                else:
                    self._restart_debounce_timer(
                        "timestep", _TIMESTEP_DEBOUNCE_S, lambda: self._apply_timestep(pace=False)
                    )

            @gui_frame_nav.on_click
            def _on_frame_nav(_) -> None:
                # Button groups cannot be disabled at runtime (viser asserts), so
                # frame stepping is ignored while playback advances on its own.
                if self._gui_playing.value:
                    return
                delta = 1 if gui_frame_nav.value == "Next Frame" else -1
                self._gui_timestep.value = (self._gui_timestep.value + delta) % num_frames

            @gui_scene_nav.on_click
            def _on_scene_nav(_) -> None:
                delta = 1 if gui_scene_nav.value == "Next Scene" else -1
                self._request_scene((self._scene_index + delta) % self._num_scenes)

            if gui_scene is not None:

                @gui_scene.on_update
                def _on_scene_slider(_) -> None:
                    if gui_scene.value == self._scene_index:
                        return
                    # Scene switches are destructive; debounce so typing a multi-digit
                    # scene number does not switch through every intermediate value.
                    self._restart_debounce_timer(
                        "scene", _SCENE_DEBOUNCE_S, lambda: self._request_scene(gui_scene.value)
                    )

            @self._gui_playing.on_update
            def _on_playing_changed(_) -> None:
                self._gui_timestep.disabled = self._gui_playing.value
                self._config.is_playing = self._gui_playing.value

            @gui_speed_options.on_click
            def _on_speed_preset(_) -> None:
                self._gui_speed.value = float(gui_speed_options.value)
                self._config.speed = self._gui_speed.value

    def _apply_timestep(self, pace: bool) -> None:
        """Run the iteration-changed callback for the current timestep value."""
        if self._on_iteration_changed is None or self._gui_timestep is None:
            return
        start = time.perf_counter()
        use_atomic = self._gui_atomic is not None and self._gui_atomic.value
        clients = list(self._server.get_clients().values()) if use_atomic else []
        with contextlib.ExitStack() as stack:
            if use_atomic:
                # server.atomic() only holds the broadcast buffer; camera updates
                # (ego follow) travel through the per-client buffers and would be
                # sent before the bulky scene data. Hold every buffer so the whole
                # timestep releases at once.
                stack.enter_context(self._server.atomic())
                for client in clients:
                    stack.enter_context(client.atomic())
            self._on_iteration_changed(self._gui_timestep.value)
        if use_atomic:
            self._server.flush()
            for client in clients:
                client.flush()

        if pace and not self._rendering:
            rendering_time = time.perf_counter() - start
            target_frame_time = self._iteration_duration_s / self._gui_speed.value
            sleep_time = target_frame_time - rendering_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _restart_debounce_timer(self, name: str, delay_s: float, action: Callable[[], None]) -> None:
        """(Re)start a named one-shot timer; only the last value before the delay expires is applied."""

        def _run() -> None:
            try:
                action()
            except Exception:
                logger.exception("Debounced %s update failed.", name)

        with self._debounce_lock:
            timer = self._debounce_timers.get(name)
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(delay_s, _run)
            timer.daemon = True
            self._debounce_timers[name] = timer
            timer.start()

    def _cancel_debounce_timers(self) -> None:
        with self._debounce_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()

    def _request_scene(self, scene_index: int) -> None:
        """Request a switch to the given scene and stop the playback loop."""
        self._requested_scene_index = scene_index
        self._should_stop = True

    def run_loop(self) -> None:
        """Blocking playback loop. Returns when the user clicks Next Scene."""
        num_frames = self._context.num_frames
        while not self._should_stop:
            if self._gui_playing.value and not self._rendering:
                self._gui_timestep.value = (self._gui_timestep.value + 1) % num_frames
            else:
                time.sleep(0.1)
        self._cancel_debounce_timers()

    def stop(self) -> None:
        """Signal the playback loop to exit."""
        self._should_stop = True
