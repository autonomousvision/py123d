import time

import numpy as np
import trimesh
import viser

from d123.common.visualization.viser.utils import (
    get_bounding_box_meshes,
    get_map_meshes,
)
from d123.dataset.scene.abstract_scene import AbstractScene

# TODO: Try to fix performance issues.
# TODO: Refactor this file.


class ViserVisualizationServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        label: str = "D123 Viser Server",
    ):

        self.server = viser.ViserServer(host=host, port=port, label=label)

    def set_scene(self, scene: AbstractScene) -> None:
        num_frames = scene.get_number_of_iterations()

        self.server.gui.configure_theme(control_width="large")
        with self.server.gui.add_folder("Playback"):

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
                meshes = []
                for name, mesh in get_bounding_box_meshes(scene, gui_timestep.value).items():
                    meshes.append(mesh)
                self.server.scene.add_mesh_trimesh(
                    f"/frame{gui_timestep.value}/detections",
                    trimesh.util.concatenate(meshes),
                    visible=True,
                )
                gui_image_handle.image = np.array(scene.get_front_cam_demo(gui_timestep.value))

                # camera_pose = _get_camera_pose_demo(scene, gui_timestep.value)
                # frustum_handle.position = camera_pose.point_3d.array
                # frustum_handle.wxyz = euler_to_quaternion_scipy(camera_pose.roll, camera_pose.pitch, camera_pose.yaw)
                # frustum_handle.image = np.array(scene.get_front_cam_demo(gui_timestep.value))

                # ego_frame_pose = _get_ego_frame_pose(scene, gui_timestep.value)
                # ego_frame_handle.position = ego_frame_pose.point_3d.array
                # ego_frame_handle.wxyz = euler_to_quaternion_scipy(ego_frame_pose.roll, ego_frame_pose.pitch, ego_frame_pose.yaw)

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

            with self.server.gui.add_folder("Camera"):
                gui_image_handle = self.server.gui.add_image(
                    image=np.array(scene.get_front_cam_demo(gui_timestep.value)),
                    label="front_cam_demo",
                    format="jpeg",
                )

            for name, mesh in get_map_meshes(scene).items():
                self.server.scene.add_mesh_trimesh(f"/map/{name}", mesh, visible=True)

            # camera_pose = _get_camera_pose_demo(scene, gui_timestep.value)
            # frustum_handle = self.server.scene.add_camera_frustum(
            #     "camera_frustum",
            #     fov=0.6724845869242845,
            #     aspect=16 / 9,
            #     scale=0.30,
            #     image=np.array(scene.get_front_cam_demo(gui_timestep.value)),
            #     position=camera_pose.point_3d.array,
            #     wxyz=euler_to_quaternion_scipy(camera_pose.roll, camera_pose.pitch, camera_pose.yaw),
            # )

            # ego_frame_pose = _get_ego_frame_pose(scene, gui_timestep.value)
            # ego_frame_handle = self.server.scene.add_frame(
            #     "ego_frame_handle",
            #     position=ego_frame_pose.point_3d.array,
            #     wxyz=euler_to_quaternion_scipy(ego_frame_pose.roll, ego_frame_pose.pitch, ego_frame_pose.yaw)
            # )

            # Playback update loop.
            prev_timestep = gui_timestep.value
            while True:
                # Update the timestep if we're playing.
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % num_frames

                # Update point size of both this timestep and the next one! There's
                # redundancy here, but this will be optimized out internally by viser.
                #
                # We update the point size for the next timestep so that it will be
                # immediately available when we toggle the visibility.
                # dynamic_meshes[gui_timestep.value].point_size = gui_point_size.value
                # dynamic_meshes[(gui_timestep.value + 1) % num_frames].point_size = gui_point_size.value
