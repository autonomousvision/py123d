import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from d123.conversion.abstract_dataset_converter import AbstractDatasetConverter
from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.conversion.datasets.pandaset.pandaset_constants import (
    PANDASET_BOX_DETECTION_FROM_STR,
    PANDASET_BOX_DETECTION_TO_DEFAULT,
    PANDASET_CAMERA_MAPPING,
    PANDASET_LOG_NAMES,
    PANDASET_SPLITS,
)
from d123.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from d123.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from d123.datatypes.detections.detection import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from d123.datatypes.scene.scene_metadata import LogMetadata
from d123.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from d123.datatypes.sensors.lidar.lidar import LiDARMetadata, LiDARType
from d123.datatypes.time.time_point import TimePoint
from d123.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from d123.datatypes.vehicle_state.vehicle_parameters import (
    get_pandaset_chrysler_pacifica_parameters,
    rear_axle_se3_to_center_se3,
)
from d123.geometry import BoundingBoxSE3, StateSE3, Vector3D
from d123.geometry.geometry_index import BoundingBoxSE3Index, EulerAnglesIndex
from d123.geometry.transform.transform_se3 import (
    convert_absolute_to_relative_se3_array,
    translate_se3_along_body_frame,
)
from d123.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from d123.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array


class PandasetConverter(AbstractDatasetConverter):
    def __init__(
        self,
        splits: List[str],
        pandaset_data_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
        train_log_names: List[str],
        val_log_names: List[str],
        test_log_names: List[str],
    ) -> None:
        super().__init__(dataset_converter_config)
        for split in splits:
            assert split in PANDASET_SPLITS, f"Split {split} is not available. Available splits: {PANDASET_SPLITS}"

        self._splits: List[str] = splits
        self._pandaset_data_root: Path = Path(pandaset_data_root)

        self._train_log_names: List[str] = train_log_names
        self._val_log_names: List[str] = val_log_names
        self._test_log_names: List[str] = test_log_names
        self._log_paths_and_split: Dict[str, List[Path]] = self._collect_log_paths()

    def _collect_log_paths(self) -> Dict[str, List[Path]]:
        log_paths_and_split: List[Tuple[Path, str]] = []

        for log_folder in self._pandaset_data_root.iterdir():
            if not log_folder.is_dir():
                continue

            log_name = log_folder.name
            assert log_name in PANDASET_LOG_NAMES, f"Log name {log_name} is not recognized."
            if (log_name in self._train_log_names) and ("pandaset_train" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_train"))
            elif (log_name in self._val_log_names) and ("pandaset_val" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_val"))
            elif (log_name in self._test_log_names) and ("pandaset_test" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_test"))

        return log_paths_and_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return 0  # NOTE: Pandaset does not have maps.

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        return None  # NOTE: Pandaset does not have maps.

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[log_index]

        # 1. Initialize Metadata
        log_metadata = LogMetadata(
            dataset="pandaset",
            split=split,
            log_name=source_log_path.name,
            location=None,  # TODO: Add location information.
            timestep_seconds=0.1,
            vehicle_parameters=get_pandaset_chrysler_pacifica_parameters(),
            camera_metadata=_get_pandaset_camera_metadata(source_log_path),
            lidar_metadata=_get_pandaset_lidar_metadata(source_log_path),
            map_metadata=None,  # NOTE: Pandaset does not have maps.
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:

            # Read files from pandaset
            timesteps = _read_json(source_log_path / "meta" / "timestamps.json")
            gps: List[Dict[str, float]] = _read_json(source_log_path / "meta" / "gps.json")
            lidar_poses: List[Dict[str, Dict[str, float]]] = _read_json(source_log_path / "lidar" / "poses.json")
            camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]] = {
                camera_name: _read_json(source_log_path / "camera" / camera_name / "poses.json")
                for camera_name in PANDASET_CAMERA_MAPPING.keys()
            }

            # Write data to log writer
            for iteration, timestep_s in enumerate(timesteps):

                ego_state = _extract_pandaset_sensor_ego_state(gps[iteration], lidar_poses[iteration])
                log_writer.write(
                    timestamp=TimePoint.from_s(timestep_s),
                    ego_state=ego_state,
                    box_detections=_extract_pandaset_box_detections(source_log_path, iteration, ego_state),
                    cameras=_extract_pandaset_sensor_camera(
                        source_log_path,
                        iteration,
                        ego_state,
                        camera_poses,
                        self.dataset_converter_config,
                    ),
                )

        # 4. Finalize log writing
        log_writer.close()


def _get_pandaset_camera_metadata(source_log_path: Path) -> Dict[PinholeCameraType, PinholeCameraMetadata]:

    all_cameras_folder = source_log_path / "camera"
    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = {}

    for camera_folder in all_cameras_folder.iterdir():
        camera_name = camera_folder.name

        assert camera_name in PANDASET_CAMERA_MAPPING.keys(), f"Camera name {camera_name} is not recognized."
        camera_type = PANDASET_CAMERA_MAPPING[camera_name]

        intrinsics_file = camera_folder / "intrinsics.json"
        assert intrinsics_file.exists(), f"Camera intrinsics file {intrinsics_file} does not exist."
        intrinsics_data = _read_json(intrinsics_file)

        camera_metadata[camera_type] = PinholeCameraMetadata(
            camera_type=camera_type,
            width=1920,
            height=1080,
            intrinsics=PinholeIntrinsics(
                fx=intrinsics_data["fx"],
                fy=intrinsics_data["fy"],
                cx=intrinsics_data["cx"],
                cy=intrinsics_data["cy"],
            ),
            distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        )

    return camera_metadata


def _get_pandaset_lidar_metadata(log_path: Path) -> Dict[LiDARType, LiDARMetadata]:
    # TODO: Implement
    return {}


def _extract_pandaset_sensor_ego_state(gps: Dict[str, float], lidar_pose: Dict[str, Dict[str, float]]) -> EgoStateSE3:

    rear_axle_pose = _main_lidar_to_rear_axle(
        StateSE3(
            x=lidar_pose["position"]["x"],
            y=lidar_pose["position"]["y"],
            z=lidar_pose["position"]["z"],
            qw=lidar_pose["heading"]["w"],
            qx=lidar_pose["heading"]["x"],
            qy=lidar_pose["heading"]["y"],
            qz=lidar_pose["heading"]["z"],
        )
    )
    # rear_axle_pose = translate_se3_along_body_frame(
    #     main_lidar_pose,
    #     vector_3d=Vector3D(x=-0.83, y=0.0, z=0.0),
    # )

    vehicle_parameters = get_pandaset_chrysler_pacifica_parameters()
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)

    # TODO: Add script to calculate the dynamic state from log sequence.
    dynamic_state = DynamicStateSE3(
        # velocity=Vector3D(x=gps["xvel"], y=gps["yvel"], z=0.0),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )

    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    )


def _extract_pandaset_box_detections(
    source_log_path: Path, iteration: int, ego_state_se3: EgoStateSE3
) -> BoxDetectionWrapper:

    # NOTE: The following provided quboids annotations are not stored in 123D
    # - stationary
    # - camera_used
    # - attributes.object_motion
    # - cuboids.sibling_id
    # - cuboids.sensor_id
    # - attributes.pedestrian_behavior
    # - attributes.pedestrian_age
    # - attributes.rider_status
    # https://github.com/scaleapi/pandaset-devkit/blob/59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0/README.md?plain=1#L288

    iteration_str = f"{iteration:02d}"
    cuboids_file = source_log_path / "annotations" / "cuboids" / f"{iteration_str}.pkl.gz"

    if not cuboids_file.exists():
        return BoxDetectionWrapper(box_detections=[])

    cuboid_df = _read_pkl_gz(cuboids_file)

    # Read cuboid data
    box_label_names = list(cuboid_df["label"])
    box_uuids = list(cuboid_df["uuid"])
    num_boxes = len(box_uuids)

    box_position_x = np.array(cuboid_df["position.x"], dtype=np.float64)
    box_position_y = np.array(cuboid_df["position.y"], dtype=np.float64)
    box_position_z = np.array(cuboid_df["position.z"], dtype=np.float64)
    box_points = np.stack([box_position_x, box_position_y, box_position_z], axis=-1)
    box_yaws = np.array(cuboid_df["yaw"], dtype=np.float64)

    # NOTE: Rather strange format to have dimensions.x as width, dimensions.y as length
    box_widths = np.array(cuboid_df["dimensions.x"], dtype=np.float64)
    box_lengths = np.array(cuboid_df["dimensions.y"], dtype=np.float64)
    box_heights = np.array(cuboid_df["dimensions.z"], dtype=np.float64)

    # Create se3 array for boxes (i.e. convert rotation to quaternion)
    box_euler_angles_array = np.zeros((num_boxes, len(EulerAnglesIndex)), dtype=np.float64)
    box_euler_angles_array[..., EulerAnglesIndex.ROLL] = DEFAULT_ROLL
    box_euler_angles_array[..., EulerAnglesIndex.PITCH] = DEFAULT_PITCH
    box_euler_angles_array[..., EulerAnglesIndex.YAW] = box_yaws

    box_se3_array = np.zeros((num_boxes, len(BoundingBoxSE3Index)), dtype=np.float64)
    box_se3_array[:, BoundingBoxSE3Index.XYZ] = box_points
    box_se3_array[:, BoundingBoxSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(box_euler_angles_array)
    box_se3_array[:, BoundingBoxSE3Index.EXTENT] = np.stack([box_lengths, box_widths, box_heights], axis=-1)

    # Fill bounding box detections and return
    box_detections: List[BoxDetectionSE3] = []
    for box_idx in range(num_boxes):
        pandaset_box_detection_type = PANDASET_BOX_DETECTION_FROM_STR[box_label_names[box_idx]]
        box_detection_type = PANDASET_BOX_DETECTION_TO_DEFAULT[pandaset_box_detection_type]

        # Convert coordinates to ISO 8855
        # NOTE: This would be faster over a batch operation.
        box_se3_array[box_idx, BoundingBoxSE3Index.STATE_SE3] = _rotate_pose_to_iso_coordinates(
            StateSE3.from_array(box_se3_array[box_idx, BoundingBoxSE3Index.STATE_SE3], copy=False)
        ).array

        box_detection_se3 = BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                detection_type=box_detection_type,
                track_token=box_uuids[box_idx],
            ),
            bounding_box_se3=BoundingBoxSE3.from_array(box_se3_array[box_idx]),
            velocity=Vector3D(0.0, 0.0, 0.0),  # TODO: Add velocity
        )
        box_detections.append(box_detection_se3)

    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_pandaset_sensor_camera(
    source_log_path: Path,
    iteration: int,
    ego_state_se3: EgoStateSE3,
    camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}
    iteration_str = f"{iteration:02d}"

    if dataset_converter_config.include_cameras:

        for camera_name, camera_type in PANDASET_CAMERA_MAPPING.items():
            image_rel_path = f"camera/{camera_name}/{iteration_str}.jpg"

            image_abs_path = source_log_path / image_rel_path
            assert image_abs_path.exists(), f"Camera image file {str(image_abs_path)} does not exist."

            camera_pose_dict = camera_poses[camera_name][iteration]
            camera_extrinsic = _rotate_pose_to_iso_coordinates(
                StateSE3(
                    x=camera_pose_dict["position"]["x"],
                    y=camera_pose_dict["position"]["y"],
                    z=camera_pose_dict["position"]["z"],
                    qw=camera_pose_dict["heading"]["w"],
                    qx=camera_pose_dict["heading"]["x"],
                    qy=camera_pose_dict["heading"]["y"],
                    qz=camera_pose_dict["heading"]["z"],
                )
            )
            # camera_extrinsic = StateSE3(
            #     x=camera_pose_dict["position"]["x"],
            #     y=camera_pose_dict["position"]["y"],
            #     z=camera_pose_dict["position"]["z"],
            #     qw=camera_pose_dict["heading"]["w"],
            #     qx=camera_pose_dict["heading"]["x"],
            #     qy=camera_pose_dict["heading"]["y"],
            #     qz=camera_pose_dict["heading"]["z"],
            # )
            camera_extrinsic = StateSE3.from_array(
                convert_absolute_to_relative_se3_array(ego_state_se3.rear_axle_se3, camera_extrinsic.array), copy=True
            )

            camera_data = None
            if dataset_converter_config.camera_store_option == "path":
                camera_data = str(image_rel_path)
            elif dataset_converter_config.camera_store_option == "binary":
                with open(image_abs_path, "rb") as f:
                    camera_data = f.read()
            camera_dict[camera_type] = camera_data, camera_extrinsic

    return camera_dict


def _extract_lidar(lidar_pc, dataset_converter_config: DatasetConverterConfig) -> Dict[LiDARType, Optional[str]]:
    # TODO: Implement this function to extract lidar data.
    return {}


def _read_json(json_file: Path):
    """Helper function to read a json file as dict."""
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def _read_pkl_gz(pkl_gz_file: Path):
    """Helper function to read a pkl.gz file as dict."""
    with gzip.open(pkl_gz_file, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def _rotate_pose_to_iso_coordinates(pose: StateSE3) -> StateSE3:
    """Helper function for pandaset to rotate a pose to ISO coordinate system (x: forward, y: left, z: up).

    NOTE: Pandaset uses a different coordinate system (x: right, y: forward, z: up).
    [1] https://arxiv.org/pdf/2112.12610.pdf

    :param pose: The input pose.
    :return: The rotated pose.
    """
    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old -X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    # F = np.eye(3, dtype=np.float64)
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    # transformation_matrix[0, 3] = pose.y
    # transformation_matrix[1, 3] = -pose.x
    # transformation_matrix[2, 3] = pose.z

    return StateSE3.from_transformation_matrix(transformation_matrix)


def _main_lidar_to_rear_axle(pose: StateSE3) -> StateSE3:

    F = np.array(
        [
            [0.0, 1.0, 0.0],  # new X = old Y (forward)
            [-1.0, 0.0, 0.0],  # new Y = old X (left)
            [0.0, 0.0, 1.0],  # new Z = old Z (up)
        ],
        dtype=np.float64,
    ).T
    # F = np.eye(3, dtype=np.float64)
    transformation_matrix = pose.transformation_matrix.copy()
    transformation_matrix[0:3, 0:3] = transformation_matrix[0:3, 0:3] @ F

    rotated_pose = StateSE3.from_transformation_matrix(transformation_matrix)

    imu_pose = translate_se3_along_body_frame(
        rotated_pose,
        vector_3d=Vector3D(x=-0.840, y=0.0, z=0.0),
    )

    # transformation_matrix[0, 3] = pose.y
    # transformation_matrix[1, 3] = -pose.x
    # transformation_matrix[2, 3] = pose.z

    return imu_pose
