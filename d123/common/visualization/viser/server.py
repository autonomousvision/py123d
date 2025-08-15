import time
from typing import Dict, List, Literal

import trimesh
import viser

from d123.common.datatypes.sensor.camera import CameraType
from d123.common.datatypes.sensor.lidar import LiDARType
from d123.common.visualization.viser.utils import (
    get_bounding_box_meshes,
    get_camera_values,
    get_lidar_points,
    get_map_meshes,
)
from d123.common.visualization.viser.utils_v2 import get_bounding_box_outlines
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

# MISC config:
LINE_WIDTH: float = 4.0

# Bounding box config:
BOUNDING_BOX_TYPE: Literal["mesh", "lines"] = "mesh"

# Map config:
MAP_AVAILABLE: bool = True


# Cameras config:

VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = [CameraType.CAM_F0, CameraType.CAM_L0, CameraType.CAM_R0]
# VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = all_camera_types
# VISUALIZE_CAMERA_FRUSTUM: List[CameraType] = []
VISUALIZE_CAMERA_GUI: List[CameraType] = []
CAMERA_SCALE: float = 2.0

# Lidar config:
LIDAR_AVAILABLE: bool = False

LIDAR_TYPES: List[LiDARType] = [
    LiDARType.LIDAR_MERGED,
    LiDARType.LIDAR_TOP,
    LiDARType.LIDAR_FRONT,
    LiDARType.LIDAR_SIDE_LEFT,
    LiDARType.LIDAR_SIDE_RIGHT,
    LiDARType.LIDAR_BACK,
]
# LIDAR_TYPES: List[LiDARType] = [
#     LiDARType.LIDAR_TOP,
# ]
LIDAR_POINT_SIZE: float = 0.05


class ViserVisualizationServer:
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
        # print(scene.available_camera_types)

        self.server.gui.configure_theme(dark_mode=False, control_width="large")

        # TODO: Fix lighting. Environment map can help, but cannot be freely configured.
        # self.server.scene.configure_environment_map(
        #     hdri="warehouse",
        #     background=False,
        #     background_intensity=0.25,
        #     environment_intensity=0.5,
        # )

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

            prev_timestep = gui_timestep.value

            # Toggle frame visibility when the timestep slider changes.
            @gui_timestep.on_update
            def _(_) -> None:
                nonlocal current_frame_handle, current_frame_handle, prev_timestep
                current_timestep = gui_timestep.value

                start = time.time()
                # with self.server.atomic():
                mew_frame_handle = self.server.scene.add_frame(f"/frame{gui_timestep.value}", show_axes=False)
                if BOUNDING_BOX_TYPE == "mesh":
                    meshes = []
                    for _, mesh in get_bounding_box_meshes(scene, gui_timestep.value).items():
                        meshes.append(mesh)
                    self.server.scene.add_mesh_trimesh(
                        f"/frame{gui_timestep.value}/detections",
                        trimesh.util.concatenate(meshes),
                        visible=True,
                    )
                elif BOUNDING_BOX_TYPE == "lines":
                    lines, colors = get_bounding_box_outlines(scene, gui_timestep.value)
                    self.server.scene.add_line_segments(
                        f"/frame{gui_timestep.value}/detections",
                        points=lines,
                        colors=colors,
                        line_width=LINE_WIDTH,
                    )
                else:
                    raise ValueError(f"Unknown bounding box type: {BOUNDING_BOX_TYPE}")

                for camera_type in VISUALIZE_CAMERA_GUI:
                    if camera_type in scene.available_camera_types:
                        camera_gui_handles[camera_type].image = scene.get_camera_at_iteration(
                            gui_timestep.value, camera_type
                        ).image

                for camera_type in VISUALIZE_CAMERA_FRUSTUM:
                    if camera_type in scene.available_camera_types:
                        camera_position, camera_quaternion, camera = get_camera_values(
                            scene, camera_type, gui_timestep.value
                        )

                        camera_frustum_handles[camera_type].position = camera_position.array
                        camera_frustum_handles[camera_type].wxyz = camera_quaternion.q
                        camera_frustum_handles[camera_type].image = camera.image

                if LIDAR_AVAILABLE:
                    points, colors = get_lidar_points(scene, gui_timestep.value, LIDAR_TYPES)
                    gui_lidar.points = points
                    gui_lidar.colors = colors

                prev_timestep = current_timestep

                rendering_time = time.time() - start
                sleep_time = 1.0 / gui_framerate.value - rendering_time
                time.sleep(max(sleep_time, 0.0))
                current_frame_handle.remove()
                current_frame_handle = mew_frame_handle
                self.server.flush()  # Optional!

            # Load in frames.
            current_frame_handle = self.server.scene.add_frame(f"/frame{gui_timestep.value}", show_axes=False)
            self.server.scene.add_frame("/map", show_axes=False)

            camera_gui_handles: Dict[CameraType, viser.GuiImageHandle] = {}
            camera_frustum_handles: Dict[CameraType, viser.CameraFrustumHandle] = {}

            for camera_type in VISUALIZE_CAMERA_GUI:
                if camera_type in scene.available_camera_types:
                    with self.server.gui.add_folder(f"Camera {camera_type.serialize()}"):
                        camera_gui_handles[camera_type] = self.server.gui.add_image(
                            image=scene.get_camera_at_iteration(gui_timestep.value, camera_type).image,
                            label=camera_type.serialize(),
                            format="jpeg",
                        )

            for camera_type in VISUALIZE_CAMERA_FRUSTUM:
                if camera_type in scene.available_camera_types:
                    camera_position, camera_quaternion, camera = get_camera_values(
                        scene, camera_type, gui_timestep.value
                    )
                    camera_frustum_handles[camera_type] = self.server.scene.add_camera_frustum(
                        f"camera_frustum_{camera_type.serialize()}",
                        fov=camera.metadata.fov_y,
                        aspect=camera.metadata.aspect_ratio,
                        scale=CAMERA_SCALE,
                        image=camera.image,
                        position=camera_position.array,
                        wxyz=camera_quaternion.q,
                    )

            if LIDAR_AVAILABLE:
                points, colors = get_lidar_points(scene, gui_timestep.value, LIDAR_TYPES)
                gui_lidar = self.server.scene.add_point_cloud(
                    name="LiDAR",
                    points=points,
                    colors=colors,
                    point_size=LIDAR_POINT_SIZE,
                    point_shape="circle",
                )

            if MAP_AVAILABLE:
                for name, mesh in get_map_meshes(scene).items():
                    self.server.scene.add_mesh_trimesh(f"/map/{name}", mesh, visible=True)

                # centerlines, __, __, road_edges = get_map_lines(scene)
                # for i, centerline in enumerate(centerlines):
                # self.server.scene.add_line_segments(
                #     "/map/centerlines",
                #     centerlines,
                #     colors=[[BLACK.rgb]],
                #     line_width=LINE_WIDTH,
                # )
                # self.server.scene.add_line_segments(
                #     "/map/left_boundary",
                #     left_boundaries,
                #     colors=[[TAB_10[2].rgb]],
                #     line_width=LINE_WIDTH,
                # )
                # self.server.scene.add_line_segments(
                #     "/map/right_boundary",clear
                #     right_boundaries,
                #     colors=[[TAB_10[3].rgb]],
                #     line_width=LINE_WIDTH,
                # )
                # print(centerlines.shape, road_edges.shape)
                # self.server.scene.add_line_segments(
                #     "/map/road_edges",
                #     road_edges,
                #     colors=[[BLACK.rgb]],
                #     line_width=LINE_WIDTH,
                # )

            # Playback update loop.
            prev_timestep = gui_timestep.value
            while server_playing:
                # Update the timestep if we're playing.
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % num_frames

        self.server.flush()
        self.next()
