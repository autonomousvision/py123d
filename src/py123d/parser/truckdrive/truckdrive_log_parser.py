"""TruckDrive log parser."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    EgoStateSE3,
    LidarMetadata,
    LogMetadata,
    Timestamp,
)
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D, Vector3DIndex
from py123d.geometry.geometry_index import EulerAnglesIndex
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array
from py123d.parser.base_dataset_parser import BaseLogParser, ModalitiesSync, ParsedCamera, ParsedLidar
from py123d.parser.truckdrive.truckdrive_constants import (
    AEVA_JOINT_LIDAR_REL_PATH,
    AEVA_REFERENCE_LIDAR_FRAME,
    CAMERA_ID_MAPPING,
    DATASET_NAME,
    DEFAULT_VEHICLE_HEIGHT_M,
    DEFAULT_VEHICLE_LENGTH_M,
    DEFAULT_VEHICLE_NAME,
    DEFAULT_VEHICLE_WIDTH_M,
    DEFAULT_WHEEL_BASE_M,
    OUSTER_LIDAR_ID_MAPPING,
    VAL_SCENES,
    VEHICLE_FRAME,
    VELODYNE_FRAME,
)
from py123d.parser.truckdrive.truckdrive_labels import TRUCKDRIVE_BOX_DETECTIONS_SE3_METADATA, decode_class_id
from py123d.parser.truckdrive.utils.truckdrive_helper import (
    SyncedFile,
    TransformTree,
    build_sync_file_index,
    discover_camera_names,
    discover_ouster_names,
    discover_sync_ids,
    load_camera_calibration,
    load_gt_trajectory,
)

logger = logging.getLogger(__name__)


class TruckDriveLogParser(BaseLogParser):
    """Lightweight handle to one TruckDrive scene log."""

    def __init__(
        self,
        data_root: Path,
        scene_name: str,
        split: str,
    ) -> None:
        self._data_root = Path(data_root)
        self._scene_name = scene_name
        self._scene_dir = self._data_root / scene_name
        self._split = split
        self._transform_tree = TransformTree(self._scene_dir / "calibrations" / "calib_tf_tree_full.json")
        self._trajectory = load_gt_trajectory(self._scene_dir / "poses" / "gt_trajectory.txt")
        self._ego_metadata = self._build_ego_metadata()
        self._camera_metadata = self._build_camera_metadata()
        self._ouster_lidar_metadata = self._build_ouster_lidar_metadata()
        self._aeva_merged_metadata = LidarMergedMetadata(
            {
                LidarID.LIDAR_MERGED: LidarMetadata(
                    lidar_name="aeva_joint_lidars",
                    lidar_id=LidarID.LIDAR_MERGED,
                    lidar_to_imu_se3=self._transform_tree.lookup(AEVA_REFERENCE_LIDAR_FRAME, VEHICLE_FRAME),
                )
            }
        )
        self._aeva_index = build_sync_file_index(
            self._scene_dir / "lidar" / "aeva" / "joint_lidars" / "points",
            ["bin"],
        )
        self._camera_indices = {
            camera_name: build_sync_file_index(
                self._scene_dir / "camera" / "leopard" / camera_name / "images",
                ["jpg", "jpeg"],
            )
            for camera_name in discover_camera_names(self._scene_dir)
        }
        self._ouster_indices = {
            ouster_name: build_sync_file_index(
                self._scene_dir / "lidar" / "ouster" / ouster_name / "points",
                ["bin"],
            )
            for ouster_name in discover_ouster_names(self._scene_dir)
        }
        self._bbox_index = build_sync_file_index(
            self._scene_dir / "annotations" / "bounding_boxes",
            ["json"],
        )
        self._lidar_aeva_to_vehicle = self._transform_tree.lookup(AEVA_REFERENCE_LIDAR_FRAME, VEHICLE_FRAME)
        self._velodyne_to_vehicle = self._transform_tree.lookup(VELODYNE_FRAME, VEHICLE_FRAME)

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset=DATASET_NAME,
            split=self._split,
            log_name=self._scene_name,
            location=None,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        for sync_id in discover_sync_ids(self._scene_dir):
            trajectory_pose = self._trajectory.get(sync_id)
            if trajectory_pose is None:
                logger.debug("Skipping sync_id %s without trajectory pose.", sync_id)
                continue

            timestamp = Timestamp.from_ns(int(round(trajectory_pose.timestamp_s * 1e9)))
            ego_state = self._build_ego_state(trajectory_pose.pose, timestamp)
            modalities = [ego_state]

            aeva_file = self._aeva_index.get(sync_id)
            if aeva_file is not None:
                modalities.append(self._build_aeva_lidar(aeva_file, timestamp))

            for ouster_name, ouster_index in self._ouster_indices.items():
                ouster_file = ouster_index.get(sync_id)
                if ouster_file is not None:
                    modalities.append(self._build_ouster_lidar(ouster_name, ouster_file, timestamp))

            for camera_name, camera_index in self._camera_indices.items():
                camera_file = camera_index.get(sync_id)
                if camera_file is not None:
                    parsed_camera = self._build_camera(camera_name, camera_file, ego_state.imu_se3, timestamp)
                    if parsed_camera is not None:
                        modalities.append(parsed_camera)

            bbox_file = self._bbox_index.get(sync_id)
            if bbox_file is not None:
                modalities.append(self._build_box_detections(bbox_file, timestamp, ego_state.imu_se3))

            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)

    def _build_ego_metadata(self) -> EgoStateSE3Metadata:
        return EgoStateSE3Metadata(
            vehicle_name=DEFAULT_VEHICLE_NAME,
            width=DEFAULT_VEHICLE_WIDTH_M,
            length=DEFAULT_VEHICLE_LENGTH_M,
            height=DEFAULT_VEHICLE_HEIGHT_M,
            wheel_base=DEFAULT_WHEEL_BASE_M,
            center_to_imu_se3=PoseSE3.identity(),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    def _build_ego_state(self, lidar_aeva_to_global: PoseSE3, timestamp: Timestamp) -> EgoStateSE3:
        vehicle_to_global = rel_to_abs_se3(
            origin=lidar_aeva_to_global,
            pose_se3=self._lidar_aeva_to_vehicle.inverse,
        )
        dynamic_state = DynamicStateSE3(
            velocity=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
            acceleration=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
            angular_velocity=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
        )
        return EgoStateSE3.from_imu(
            imu_se3=vehicle_to_global,
            metadata=self._ego_metadata,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state,
        )

    def _build_camera_metadata(self) -> Dict[str, PinholeCameraMetadata]:
        metadata: Dict[str, PinholeCameraMetadata] = {}
        for camera_name in discover_camera_names(self._scene_dir):
            camera_id = CAMERA_ID_MAPPING.get(camera_name)
            if camera_id is None:
                logger.warning("No CameraID mapping for TruckDrive camera %s", camera_name)
                continue
            calib_path = self._scene_dir / "calibrations" / f"calib_camera_leopard_{camera_name}.json"
            if not calib_path.is_file():
                continue
            height, width, fx, fy, cx, cy = load_camera_calibration(calib_path)
            camera_frame = f"camera_leopard_{camera_name}"
            camera_to_vehicle = self._transform_tree.lookup(camera_frame, VEHICLE_FRAME)
            metadata[camera_name] = PinholeCameraMetadata(
                camera_name=camera_name,
                camera_id=camera_id,
                intrinsics=PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
                distortion=None,
                width=width,
                height=height,
                camera_to_imu_se3=camera_to_vehicle,
                is_undistorted=True,
            )
        return metadata

    def _build_ouster_lidar_metadata(self) -> Dict[str, LidarMetadata]:
        metadata: Dict[str, LidarMetadata] = {}
        for ouster_name in discover_ouster_names(self._scene_dir):
            lidar_id = OUSTER_LIDAR_ID_MAPPING.get(ouster_name)
            if lidar_id is None:
                continue
            ouster_frame = f"lidar_ouster_{ouster_name}"
            metadata[ouster_name] = LidarMetadata(
                lidar_name=ouster_name,
                lidar_id=lidar_id,
                lidar_to_imu_se3=self._transform_tree.lookup(ouster_frame, VEHICLE_FRAME),
            )
        return metadata

    def _build_aeva_lidar(self, synced_file: SyncedFile, timestamp: Timestamp) -> ParsedLidar:
        relative_path = Path(self._scene_name) / AEVA_JOINT_LIDAR_REL_PATH / synced_file.filename
        return ParsedLidar(
            metadata=self._aeva_merged_metadata,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            dataset_root=self._data_root,
            relative_path=relative_path,
        )

    def _build_ouster_lidar(self, ouster_name: str, synced_file: SyncedFile, timestamp: Timestamp) -> ParsedLidar:
        relative_path = (
            Path(self._scene_name) / "lidar" / "ouster" / ouster_name / "points" / synced_file.filename
        )
        return ParsedLidar(
            metadata=self._ouster_lidar_metadata[ouster_name],
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            dataset_root=self._data_root,
            relative_path=relative_path,
        )

    def _build_camera(
        self,
        camera_name: str,
        synced_file: SyncedFile,
        imu_to_global: PoseSE3,
        timestamp: Timestamp,
    ) -> Optional[ParsedCamera]:
        metadata = self._camera_metadata.get(camera_name)
        if metadata is None:
            return None
        relative_path = Path(self._scene_name) / "camera" / "leopard" / camera_name / "images" / synced_file.filename
        if not (self._data_root / relative_path).is_file():
            return None
        camera_to_global = rel_to_abs_se3(origin=imu_to_global, pose_se3=metadata.camera_to_imu_se3)
        return ParsedCamera(
            metadata=metadata,
            timestamp=timestamp,
            camera_to_global_se3=camera_to_global,
            dataset_root=self._data_root,
            relative_path=relative_path,
        )

    def _build_box_detections(self, synced_file: SyncedFile, timestamp: Timestamp, vehicle_to_global: PoseSE3) -> BoxDetectionsSE3:
        json_path = self._scene_dir / "annotations" / "bounding_boxes" / synced_file.filename
        with json_path.open("r", encoding="utf-8") as file:
            raw_objects = json.load(file)

        if isinstance(raw_objects, dict):
            objects = list(raw_objects.values())
        else:
            objects = raw_objects

        box_detections: List[BoxDetectionSE3] = []
        zero_velocity = Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64))
        velodyne_to_global = rel_to_abs_se3(origin=vehicle_to_global, pose_se3=self._velodyne_to_vehicle)
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            class_id = str(obj.get("class-id", ""))
            label = decode_class_id(class_id)
            if label is None:
                continue

            center_x = float(obj.get("x", obj.get("x_global", 0.0)))
            center_y = float(obj.get("y", obj.get("y_global", 0.0)))
            center_z = float(obj.get("z", obj.get("z_global", 0.0)))
            length = float(obj.get("l", obj.get("l_global", 0.0)))
            width = float(obj.get("w", obj.get("w_global", 0.0)))
            height = float(obj.get("h", obj.get("h_global", 0.0)))
            yaw = float(obj.get("yaw", obj.get("yaw_global", 0.0)))

            euler = np.zeros(len(EulerAnglesIndex), dtype=np.float64)
            euler[EulerAnglesIndex.YAW] = yaw
            quaternion = get_quaternion_array_from_euler_array(euler.reshape(1, -1))[0]
            center_se3_velodyne = PoseSE3(
                x=center_x,
                y=center_y,
                z=center_z,
                qw=float(quaternion[0]),
                qx=float(quaternion[1]),
                qy=float(quaternion[2]),
                qz=float(quaternion[3]),
            )
            center_se3 = rel_to_abs_se3(origin=velodyne_to_global, pose_se3=center_se3_velodyne)
            tracking_id = obj.get("Tracking_ID", -9999)
            object_id = str(obj.get("object-id", obj.get("id", "")))
            if tracking_id in (-9999, None):
                track_token = object_id or "-1"
            else:
                track_token = str(tracking_id)

            box_detections.append(
                BoxDetectionSE3(
                    attributes=BoxDetectionAttributes(label=label, track_token=track_token),
                    bounding_box_se3=BoundingBoxSE3(
                        center_se3=center_se3,
                        length=length,
                        width=width,
                        height=height,
                    ),
                    velocity_3d=zero_velocity,
                )
            )

        return BoxDetectionsSE3(
            box_detections=box_detections,
            timestamp=timestamp,
            metadata=TRUCKDRIVE_BOX_DETECTIONS_SE3_METADATA,
        )


def resolve_truckdrive_split(scene_name: str) -> str:
    """Resolve the py123d split name for a TruckDrive scene."""
    if scene_name in VAL_SCENES:
        return "truckdrive_val"
    return "truckdrive_train"
