import gc
import json
import os
import pickle
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import yaml
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
from pyquaternion import Quaternion
from sqlalchemy import func


from kitti_360_data_converter import _extract_ego_state_all,get_kitti360_lidar_metadata,_extract_cameras,_extract_detections

# a =  _extract_ego_state_all("2013_05_28_drive_0000_sync")
# print(a[0])
# print(a[1])
# print(a[10])
from d123.common.datatypes.time.time_point import TimePoint
from d123.common.datatypes.sensor.camera import CameraMetadata, CameraType, camera_metadata_dict_to_json

NUPLAN_CAMERA_TYPES = {
    CameraType.CAM_F0: CameraChannel.CAM_F0,
    CameraType.CAM_B0: CameraChannel.CAM_B0,
    CameraType.CAM_L0: CameraChannel.CAM_L0,
    CameraType.CAM_L1: CameraChannel.CAM_L1,
    CameraType.CAM_L2: CameraChannel.CAM_L2,
    CameraType.CAM_R0: CameraChannel.CAM_R0,
    CameraType.CAM_R1: CameraChannel.CAM_R1,
    CameraType.CAM_R2: CameraChannel.CAM_R2,
}

NUPLAN_DATA_ROOT = Path(os.environ["NUPLAN_DATA_ROOT"])
NUPLAN_ROLLING_SHUTTER_S: Final[TimePoint] = TimePoint.from_s(1 / 60)

def _extract_camera(
    log_db: NuPlanDB,
    lidar_pc: LidarPc,
    source_log_path: Path,
) -> Dict[CameraType, Union[str, bytes]]:

    camera_dict: Dict[str, Union[str, bytes]] = {}
    sensor_root = NUPLAN_DATA_ROOT / "nuplan-v1.1" / "sensor_blobs"

    log_cam_infos = {camera.token: camera for camera in log_db.log.cameras}
    for camera_type, camera_channel in NUPLAN_CAMERA_TYPES.items():
        camera_data: Optional[Union[str, bytes]] = None
        c2e: Optional[List[float]] = None
        image_class = list(get_images_from_lidar_tokens(source_log_path, [lidar_pc.token], [str(camera_channel.value)]))
        # print("image_class",image_class)
        if len(image_class) != 0:
            image = image_class[0]
            filename_jpg = sensor_root / image.filename_jpg

            timestamp = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us
            img_ego_pose: EgoPose = (
                log_db.log._session.query(EgoPose).order_by(func.abs(EgoPose.timestamp - timestamp)).first()
            )
            img_e2g = img_ego_pose.trans_matrix
            g2e = lidar_pc.ego_pose.trans_matrix_inv
            img_e2e = g2e @ img_e2g
            cam_info = log_cam_infos[image.camera_token]
            c2img_e = cam_info.trans_matrix
            c2e = img_e2e @ c2img_e
            # print(f"Camera {camera_type} found for lidar {lidar_pc.token} at timestamp {timestamp}")
            print(camera_type,"c2e:", c2e)
        camera_dict[camera_type] = camera_data

    return camera_dict


def get_cam_info_from_lidar_pc(log,log_file, lidar_pc, rolling_shutter_s=1/60):
    
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pc.token], [str(channel.value) for channel in CameraChannel]
    )

    # if interp_trans:
    #     neighbours = []
    #     ego_poses_dict = {}
    #     for ego_pose in log.ego_poses:
    #         ego_poses_dict[ego_pose.token] = ego_pose
    #         if abs(ego_pose.timestamp - lidar_pc.ego_pose.timestamp) / 1e6 < 0.5:
    #             neighbours.append(ego_pose)
    #     timestamps = [pose.timestamp for pose in neighbours]
    #     translations = [pose.translation_np for pose in neighbours]
    #     splines = [CubicSpline(timestamps, [translation[i] for translation in translations]) for i in range(2)]

    log_cam_infos = {camera.token : camera for camera in log.camera}
    cams = {}
    for img in retrieved_images:
        channel = img.channel
        filename = img.filename_jpg

        # if interp_trans:
            # img_ego_pose = ego_poses_dict[img.ego_pose_token]
            # interpolated_translation = np.array([splines[0](timestamp), splines[1](timestamp), img_ego_pose.z])
            # delta = interpolated_translation - lidar_pc.ego_pose.translation_np
            # delta = np.dot(lidar_pc.ego_pose.quaternion.rotation_matrix.T, delta)
        if channel == "CAM_F0":
            timestamp = img.timestamp + (rolling_shutter_s * 1e6)
            img_ego_pose = log.session.query(EgoPose).order_by(func.abs(EgoPose.timestamp - timestamp)).first()
            img_e2g = img_ego_pose.trans_matrix
            # print("img_e2g:", img_e2g)
            
            g2e = lidar_pc.ego_pose.trans_matrix_inv
            # print("g2e:", g2e)   #change obviously
            img_e2e = g2e @ img_e2g
            # print("img_e2e:", img_e2e)
            cam_info = log_cam_infos[img.camera_token]
            c2img_e = cam_info.trans_matrix
            # print("c2img_e:", c2img_e)
            c2e = img_e2e @ c2img_e
            # print("channel:", channel, "c2e:", c2e)
            
            cams[channel] = dict(
                data_path = filename,
                timestamp = img.timestamp,
                token=img.token,
                sensor2ego_rotation = Quaternion(matrix=c2e[:3, :3]),
                sensor2ego_translation = c2e[:3, 3],
                cam_intrinsic = cam_info.intrinsic_np,
                distortion = cam_info.distortion_np,
            )
        

    if len(cams) != 8:
        return None
    # print(cams)
    return cams

if __name__ == "__main__":
    # Example usage
    # data_converter_config: DataConverterConfig
    # log_path = Path("/nas/datasets/nuplan/nuplan-v1.1/splits/mini/2021.10.11.07.12.18_veh-50_00211_00304.db")
    # log_path = Path("/nas/datasets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db")
    # log_db = NuPlanDB(NUPLAN_DATA_ROOT, str(log_path), None)

    # for lidar_pc in  log_db.lidar_pc:  # Replace with actual token
    #     # camera_data = _extract_camera(log_db, lidar_pc, log_path)
    #     camera_data = get_cam_info_from_lidar_pc(log_db,log_path, lidar_pc, rolling_shutter_s=1/60)
    # print(_extract_cameras("2013_05_28_drive_0000_sync",0))
    _extract_detections("2013_05_28_drive_0000_sync", 0)