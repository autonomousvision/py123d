import concurrent.futures
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import viser

from py123d.datatypes.scene.abstract_scene import AbstractScene
from py123d.datatypes.sensors.lidar import LiDARType
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraType
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import StateSE3Index
from py123d.geometry.transform.transform_se3 import (
    convert_relative_to_absolute_points_3d_array,
    convert_relative_to_absolute_se3_array,
)
from py123d.visualization.color.color import TAB_10
from py123d.visualization.viser.viser_config import ViserConfig


def add_camera_frustums_to_viser_server(
    scene: AbstractScene,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    camera_frustum_handles: Dict[PinholeCameraType, viser.CameraFrustumHandle],
) -> None:

    if viser_config.camera_frustum_visible:
        scene_center_array = initial_ego_state.center.point_3d.array
        ego_pose = scene.get_ego_state_at_iteration(scene_interation).rear_axle_se3.array
        ego_pose[StateSE3Index.XYZ] -= scene_center_array

        def _add_camera_frustums_to_viser_server(camera_type: PinholeCameraType) -> None:
            camera = scene.get_pinhole_camera_at_iteration(scene_interation, camera_type)
            if camera is not None:
                camera_position, camera_quaternion, camera_image = _get_camera_values(
                    camera,
                    ego_pose.copy(),
                    viser_config.camera_frustum_image_scale,
                )
                if camera_type in camera_frustum_handles:
                    camera_frustum_handles[camera_type].position = camera_position
                    camera_frustum_handles[camera_type].wxyz = camera_quaternion
                    camera_frustum_handles[camera_type].image = camera_image
                else:
                    camera_frustum_handles[camera_type] = viser_server.scene.add_camera_frustum(
                        f"camera_frustums/{camera_type.serialize()}",
                        fov=camera.metadata.fov_y,
                        aspect=camera.metadata.aspect_ratio,
                        scale=viser_config.camera_frustum_frustum_scale,
                        image=camera_image,
                        position=camera_position,
                        wxyz=camera_quaternion,
                    )

            return None

        # NOTE; In order to speed up adding camera frustums, we use multithreading and resize the images.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(viser_config.camera_frustum_types)) as executor:
            future_to_camera = {
                executor.submit(_add_camera_frustums_to_viser_server, camera_type): camera_type
                for camera_type in viser_config.camera_frustum_types
            }
            for future in concurrent.futures.as_completed(future_to_camera):
                _ = future.result()

        # TODO: Remove serial implementation, if not needed anymore.
        # for camera_type in viser_config.camera_frustum_types:
        #     _add_camera_frustums_to_viser_server(camera_type)

        return None


def add_camera_gui_to_viser_server(
    scene: AbstractScene,
    scene_interation: int,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    camera_gui_handles: Dict[PinholeCameraType, viser.GuiImageHandle],
) -> None:
    if viser_config.camera_gui_visible:
        for camera_type in viser_config.camera_gui_types:
            camera = scene.get_pinhole_camera_at_iteration(scene_interation, camera_type)
            if camera is not None:
                if camera_type in camera_gui_handles:
                    camera_gui_handles[camera_type].image = _rescale_image(
                        camera.image, viser_config.camera_gui_image_scale
                    )
                else:
                    with viser_server.gui.add_folder(f"Camera {camera_type.serialize()}"):
                        camera_gui_handles[camera_type] = viser_server.gui.add_image(
                            image=_rescale_image(camera.image, viser_config.camera_gui_image_scale),
                            label=camera_type.serialize(),
                        )


def add_lidar_pc_to_viser_server(
    scene: AbstractScene,
    scene_interation: int,
    initial_ego_state: EgoStateSE3,
    viser_server: viser.ViserServer,
    viser_config: ViserConfig,
    lidar_pc_handle: Optional[viser.PointCloudHandle],
) -> None:
    if viser_config.lidar_visible:

        scene_center_array = initial_ego_state.center.point_3d.array
        ego_pose = scene.get_ego_state_at_iteration(scene_interation).rear_axle_se3.array
        ego_pose[StateSE3Index.XYZ] -= scene_center_array

        def _load_lidar_points(lidar_type: LiDARType) -> npt.NDArray[np.float32]:
            lidar = scene.get_lidar_at_iteration(scene_interation, lidar_type)
            if lidar is not None:
                return lidar.xyz
            else:
                return np.zeros((0, 3), dtype=np.float32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(viser_config.lidar_types)) as executor:
            future_to_lidar = {
                executor.submit(_load_lidar_points, lidar_type): lidar_type for lidar_type in viser_config.lidar_types
            }
            lidar_points_3d_list: List[npt.NDArray[np.float32]] = []
            for future in concurrent.futures.as_completed(future_to_lidar):
                lidar_points_3d_list.append(future.result())

        points_3d_local = (
            np.concatenate(lidar_points_3d_list, axis=0) if lidar_points_3d_list else np.zeros((0, 3), dtype=np.float32)
        )

        colors = []
        for idx, points in enumerate(lidar_points_3d_list):
            color = np.array(TAB_10[idx % len(TAB_10)].rgb, dtype=np.uint8)
            colors.append(np.tile(color, (points.shape[0], 1)))
        colors = np.vstack(colors) if colors else np.zeros((0, 3), dtype=np.uint8)

        points = convert_relative_to_absolute_points_3d_array(ego_pose, points_3d_local)
        colors = np.zeros_like(points)

        # # TODO: remove:
        # lidar = scene.get_lidar_at_iteration(scene_interation, LiDARType.LIDAR_TOP)
        # lidar_extrinsic = convert_relative_to_absolute_se3_array(
        #     origin=ego_pose, se3_array=lidar.metadata.extrinsic.array
        # )

        # viser_server.scene.add_frame(
        #     "lidar_frame",
        #     position=lidar_extrinsic[StateSE3Index.XYZ],
        #     wxyz=lidar_extrinsic[StateSE3Index.QUATERNION],
        # )

        if lidar_pc_handle is not None:
            lidar_pc_handle.points = points
            lidar_pc_handle.colors = colors
        else:
            lidar_pc_handle = viser_server.scene.add_point_cloud(
                "lidar_points",
                points=points,
                colors=colors,
                point_size=viser_config.lidar_point_size,
                point_shape=viser_config.lidar_point_shape,
            )


def _get_camera_values(
    camera: PinholeCamera,
    ego_pose: npt.NDArray[np.float64],
    resize_factor: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    assert ego_pose.ndim == 1 and len(ego_pose) == len(StateSE3Index)

    rel_camera_pose = camera.extrinsic.array
    abs_camera_pose = convert_relative_to_absolute_se3_array(origin=ego_pose, se3_array=rel_camera_pose)

    camera_position = abs_camera_pose[StateSE3Index.XYZ]
    camera_rotation = abs_camera_pose[StateSE3Index.QUATERNION]

    camera_image = _rescale_image(camera.image, resize_factor)
    return camera_position, camera_rotation, camera_image


def _rescale_image(image: npt.NDArray[np.uint8], scale: float) -> npt.NDArray[np.uint8]:
    if scale == 1.0:
        return image
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return downscaled_image
