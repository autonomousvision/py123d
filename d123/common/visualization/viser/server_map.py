import time
from typing import List, Literal

import viser

from d123.common.datatypes.sensor.camera import CameraType
from d123.common.visualization.color.color import BLACK, TAB_10
from d123.common.visualization.viser.utils import get_map_lines, get_map_meshes
from d123.dataset.scene.abstract_scene import AbstractScene

# TODO: Try to fix performance issues.
# TODO: Refactor this file.

all_camera_types: List[CameraType] = [
    CameraType.CAM_F0,
    CameraType.CAM_B0,
    CameraType.CAM_L0,
    CameraType.CAM_L1,
    CameraType.CAM_L2,
    CameraType.CAM_R0,
    CameraType.CAM_R1,
    CameraType.CAM_R2,
]

LIDAR_POINT_SIZE: float = 0.05
MAP_AVAILABLE: bool = True
BOUNDING_BOX_TYPE: Literal["mesh", "lines"] = "lines"
LINE_WIDTH: float = 4.0

CAMERA_SCALE: float = 1.0

# VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = [
#     CameraType.CAM_F0,
#     CameraType.CAM_L0,
#     CameraType.CAM_R0,
#     CameraType.CAM_L1,
#     CameraType.CAM_R1,
# ]
# VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = [CameraType.CAM_F0, CameraType.CAM_L0, CameraType.CAM_R0]
# VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = all_camera_types
VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = []
VISUALIZE_CAMERA_GUI: List[CameraType] = [CameraType.CAM_F0]
LIDAR_AVAILABLE: bool = False


class ViserMapVisualizationServer:
    def __init__(
        self,
        scenes: List[AbstractScene],
        scene_index: int = 0,
        host: str = "localhost",
        port: int = 8080,
        label: str = "D123 Viser Server",
    ):
        assert len(scenes) > 0, "At least one scene must be provided."
        self.scenes = scenes
        self.scene_index = scene_index

        self.host = host
        self.port = port
        self.label = label

        self.server = viser.ViserServer(host=self.host, port=self.port, label=self.label)
        self.set_scene(self.scenes[self.scene_index % len(self.scenes)])

    def next(self) -> None:
        self.server.flush()
        self.server.gui.reset()
        self.server.scene.reset()
        self.scene_index = (self.scene_index + 1) % len(self.scenes)
        print(f"Viser server started at {self.host}:{self.port}")
        self.set_scene(self.scenes[self.scene_index])

    def set_scene(self, scene: AbstractScene) -> None:
        num_frames = scene.get_number_of_iterations()
        print(scene.available_camera_types)

        self.server.gui.configure_theme(control_width="large")
        with self.server.gui.add_folder("Playback"):
            server_playing = True

            gui_timestep = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            gui_next_scene = self.server.gui.add_button("Next Scene", disabled=False)
            gui_playing = self.server.gui.add_checkbox("Playing", True)
            gui_framerate = self.server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)
            gui_framerate_options = self.server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))

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

                # Frame step buttons

            # Toggle frame visibility when the timestep slider changes.
            @gui_timestep.on_update
            def _(_) -> None:

                time.sleep(0.1)

            prev_timestep = gui_timestep.value

            # Load in frames.
            if MAP_AVAILABLE:
                for name, mesh in get_map_meshes(scene).items():
                    self.server.scene.add_mesh_trimesh(f"/map/{name}", mesh, visible=True)

                centerlines, left_boundaries, right_boundaries = get_map_lines(scene)
                # for i, centerline in enumerate(centerlines):
                self.server.scene.add_line_segments(
                    "/map/centerlines",
                    centerlines,
                    colors=[[BLACK.rgb]],
                    line_width=LINE_WIDTH,
                )
                self.server.scene.add_line_segments(
                    "/map/left_boundary",
                    left_boundaries,
                    colors=[[TAB_10[2].rgb]],
                    line_width=LINE_WIDTH,
                )
                self.server.scene.add_line_segments(
                    "/map/right_boundary",
                    right_boundaries,
                    colors=[[TAB_10[3].rgb]],
                    line_width=LINE_WIDTH,
                )

            # Playback update loop.
            prev_timestep = gui_timestep.value
            while server_playing:
                # Update the timestep if we're playing.
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % num_frames

        self.server.flush()
        self.next()
