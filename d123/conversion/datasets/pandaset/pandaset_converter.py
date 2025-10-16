import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from d123.conversion.abstract_dataset_converter import AbstractDatasetConverter
from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.conversion.datasets.pandaset.pandaset_constants import (
    PANDASET_CAMERA_MAPPING,
    PANDASET_LOG_NAMES,
    PANDASET_SPLITS,
)
from d123.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from d123.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from d123.datatypes.detections.detection import BoxDetectionWrapper
from d123.datatypes.maps.map_metadata import MapMetadata
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
from d123.geometry import StateSE3, Vector3D


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
                log_paths_and_split.append((log_folder, "train"))
            elif (log_name in self._val_log_names) and ("pandaset_val" in self._splits):
                log_paths_and_split.append((log_folder, "val"))
            elif (log_name in self._test_log_names) and ("pandaset_test" in self._splits):
                log_paths_and_split.append((log_folder, "test"))

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

            timesteps = _read_json(source_log_path / "meta" / "timestamps.json")
            gps: List[Dict[str, float]] = _read_json(source_log_path / "meta" / "gps.json")
            lidar_poses: List[Dict[str, Dict[str, float]]] = _read_json(source_log_path / "lidar" / "poses.json")

            for iteration, timestep_s in enumerate(timesteps):
                iteration_str = f"{iteration:02d}"

                ego_state = _extract_pandaset_sensor_ego_state(gps[iteration], lidar_poses[iteration])
                log_writer.write(
                    timestamp=TimePoint.from_s(timestep_s),
                    ego_state=ego_state,
                    box_detections=_extract_pandaset_sensor_box_detections(source_log_path, iteration_str, ego_state),
                    cameras=_extract_pandaset_sensor_camera(self.dataset_converter_config),
                )

        # 4. Finalize log writing
        log_writer.close()


def _get_pandaset_sensor_map_metadata(split: str, log_name: str) -> MapMetadata:
    return MapMetadata(
        dataset="pandaset-sensor",
        split=split,
        log_name=log_name,
        location=None,  # TODO: Add location information, e.g. city name.
        map_has_z=True,
        map_is_local=True,
    )


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


def _extract_pandaset_sensor_box_detections(
    source_log_path: Path,
    iteration_str: str,
    ego_state_se3: EgoStateSE3,
) -> BoxDetectionWrapper:

    # TODO: Implement

    cuboids_file = source_log_path / "annotations" / "cuboids" / f"{iteration_str}.pkl.gz"

    if not cuboids_file.exists():
        return BoxDetectionWrapper(box_detections=[])

    # cuboid_df = _read_pkl_gz(cuboids_file)

    # labels = list(cuboid_df["label"])
    # position_x = list(cuboid_df["position_x"])
    # position_y = list(cuboid_df["position_y"])
    # position_z = list(cuboid_df["position_z"])
    # yaws = list(cuboid_df["yaw"])

    # annotations_slice = _get_pandaset_camera_metadata(annotations_df, lidar_timestamp_ns)
    # num_detections = len(annotations_slice)

    # detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    # detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    # detections_token: List[str] = annotations_slice["track_uuid"].tolist()
    # detections_types: List[DetectionType] = []

    # for detection_idx, (_, row) in enumerate(annotations_slice.iterrows()):
    #     row = row.to_dict()

    #     detections_state[detection_idx, BoundingBoxSE3Index.XYZ] = [row["tx_m"], row["ty_m"], row["tz_m"]]
    #     detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = [row["qw"], row["qx"], row["qy"], row["qz"]]
    #     detections_state[detection_idx, BoundingBoxSE3Index.EXTENT] = [row["length_m"], row["width_m"], row["height_m"]]

    #     pandaset_detection_type = PANDASET_BOX_DETECTION_MAPPING.deserialize(row["category"])
    #     detections_types.append(PANDASET_BOX_DETECTION_MAPPING[pandaset_detection_type])

    # detections_state[:, BoundingBoxSE3Index.STATE_SE3] = convert_relative_to_absolute_se3_array(
    #     origin=ego_state_se3.rear_axle_se3,
    #     se3_array=detections_state[:, BoundingBoxSE3Index.STATE_SE3],
    # )

    # box_detections: List[BoxDetectionSE3] = []
    # for detection_idx in range(num_detections):
    #     box_detections.append(
    #         BoxDetectionSE3(
    #             metadata=BoxDetectionMetadata(
    #                 detection_type=detections_types[detection_idx],
    #                 timepoint=None,
    #                 track_token=detections_token[detection_idx],
    #                 confidence=None,
    #             ),
    #             bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
    #             velocity=Vector3D.from_array(detections_velocity[detection_idx]),
    #         )
    #     )

    # return BoxDetectionWrapper(box_detections=box_detections)
    return BoxDetectionWrapper(box_detections=[])


def _extract_pandaset_sensor_ego_state(gps: Dict[str, float], lidar_pose: Dict[str, Dict[str, float]]) -> EgoStateSE3:

    rear_axle_pose = StateSE3(
        x=lidar_pose["position"]["x"],
        y=lidar_pose["position"]["y"],
        z=lidar_pose["position"]["z"],
        qw=lidar_pose["heading"]["w"],
        qx=lidar_pose["heading"]["x"],
        qy=lidar_pose["heading"]["y"],
        qz=lidar_pose["heading"]["z"],
    )

    vehicle_parameters = get_pandaset_chrysler_pacifica_parameters()
    center = rear_axle_se3_to_center_se3(rear_axle_se3=rear_axle_pose, vehicle_parameters=vehicle_parameters)

    # TODO: Add script to calculate the dynamic state from log sequence.
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(x=gps["xvel"], y=gps["yvel"], z=gps["zvel"]),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )

    return EgoStateSE3(
        center_se3=center,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,
    )


def _extract_pandaset_sensor_camera(
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]]:

    # TODO: Implement

    # camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes], StateSE3]] = {}
    # split = source_log_path.parent.name
    # log_id = source_log_path.name

    # source_dataset_dir = source_log_path.parent.parent

    # for _, row in egovehicle_se3_sensor_df.iterrows():
    #     row = row.to_dict()
    #     if row["sensor_name"] not in PANDASET_CAMERA_MAPPING:
    #         continue

    #     camera_name = row["sensor_name"]
    #     camera_type = PANDASET_CAMERA_MAPPING[camera_name]

    #     relative_image_path = find_closest_target_fpath(
    #         split=split,
    #         log_id=log_id,
    #         src_sensor_name="lidar",
    #         src_timestamp_ns=lidar_timestamp_ns,
    #         target_sensor_name=camera_name,
    #         synchronization_df=synchronization_df,
    #     )
    #     if relative_image_path is not None:
    #         absolute_image_path = source_dataset_dir / relative_image_path
    #         assert absolute_image_path.exists()

    #         # TODO: Adjust for finer IMU timestamps to correct the camera extrinsic.
    #         camera_extrinsic = StateSE3(
    #             x=row["tx_m"],
    #             y=row["ty_m"],
    #             z=row["tz_m"],
    #             qw=row["qw"],
    #             qx=row["qx"],
    #             qy=row["qy"],
    #             qz=row["qz"],
    #         )
    #         camera_data = None
    #         if dataset_converter_config.camera_store_option == "path":
    #             camera_data = str(relative_image_path)
    #         elif dataset_converter_config.camera_store_option == "binary":
    #             with open(absolute_image_path, "rb") as f:
    #                 camera_data = f.read()
    #         camera_dict[camera_type] = camera_data, camera_extrinsic

    # return camera_dict
    return {}


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
