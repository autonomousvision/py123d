"""
NUREC Dataset Converter.

Based on the data I downloaded from HuggingFace, this is the current data
structure.
QUESTION / TODO : Should we keep the reading more flexible or can we assume a
fixed structure?

As I couldn't find clear documentation on some information, I either not
included them, just guessed what they could be, or queried a **LLM** about what
it could be.

Except for the obvious refactoring, there are still two large TODO's IMO:
1. Currently the street is above the objects
2. The map doesn't align with the provided poses of the objects, thus the cars
are driving not on the streets.

    datapath/
        Batch0002/
            scene_hash_1/
                camera_description.mp4  # e.g. camera_front_wide_120fov.mp4
                labels.json
                scene_hash_1.usdz
                EXTRACTED/  # Extracted from .usdz
                    map.xodr
                    datasource_summary.json
                    sequence_tracks.json
                    pose_record.json
                    data_info.json
                    checkpoint.ckpt
                    volume.nurec
                    mesh*.ply/usd
            scene_hash_2/
            ...
        Batch0003/
        Batch0008/
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.nurec.utils.nurec_constants import (
    NUREC_CAMERA_MAPPING,
    NUREC_DEFAULT_BOX_DIMENSIONS,
    NUREC_TARGET_DT,
    NUREC_TO_DETECTION_TYPE,
)
from py123d.conversion.datasets.nurec.utils.nurec_helpers import (
    extract_video_frame,
    find_nearest_camera_frame_index,
    find_nearest_timestamp_index,
    get_available_camera,
    get_video_info,
    interpolate_pose,
    load_json,
    transformation_matrix_to_pose_7d,
)
from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter
from py123d.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.utils.map_utils.opendrive.opendrive_map_conversion import convert_xodr_map
from py123d.datatypes.detections.detection import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from py123d.datatypes.detections.detection_types import DetectionType
from py123d.datatypes.maps.map_metadata import MapMetadata
from py123d.datatypes.scene.scene_metadata import LogMetadata
from py123d.datatypes.sensors.camera.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeCameraType,
    PinholeDistortion,
    PinholeIntrinsics,
)
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import BoundingBoxSE3, StateSE3, Vector3D


class NURECConverter(AbstractDatasetConverter):
    """Converter for NUREC dataset."""

    def __init__(
        self,
        splits: List[str],
        nurec_data_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)

        self._splits: List[str] = splits
        self._nurec_data_root: Path = Path(nurec_data_root)

        # Collect all scene paths organized by split
        self._split_scene_path_pairs: List[Tuple[str, Path, str]] = self._collect_split_scene_path_pairs()

    def _collect_split_scene_path_pairs(self) -> List[Tuple[str, Path, str]]:
        """
        Collect (split, scene_path, batch_name) tuples for all scenes.

        Returns:
            List of tuples containing split name, scene path, and batch name
        """
        split_scene_path_pairs: List[Tuple[str, Path, str]] = []

        for split in self._splits:
            # Extract batch name from split (e.g., "nurec_batch0002" -> "Batch0002")
            batch_name = "Batch" + split.split("batch")[-1]
            batch_path = self._nurec_data_root / batch_name

            if not batch_path.exists():
                print(f"Warning: Batch path {batch_path} does not exist, skipping.")
                continue

            # Get all scene directories in this batch
            scene_dirs = [d for d in batch_path.iterdir() if d.is_dir()]

            for scene_dir in scene_dirs:
                # Extract USDZ if not already extracted
                usdz_file = scene_dir / f"{scene_dir.name}.usdz"
                extracted_path = scene_dir / "EXTRACTED"

                if usdz_file.exists() and not extracted_path.exists():
                    print(f"Extracting {usdz_file}...")
                    with zipfile.ZipFile(usdz_file, "r") as zip_ref:
                        zip_ref.extractall(extracted_path)

                split_scene_path_pairs.append((split, scene_dir, batch_name))

        return split_scene_path_pairs

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        # Each scene has its own map
        return len(self._split_scene_path_pairs)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        # Each scene is a separate log
        return len(self._split_scene_path_pairs)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        split, scene_path, batch_name = self._split_scene_path_pairs[map_index]

        # Initialize map metadata
        map_metadata = MapMetadata(
            dataset="nurec",
            split=split,
            log_name=scene_path.name,
            location=scene_path.name,  # Use scene hash as location
            map_has_z=True,
            map_is_local=True,  # nurec maps are in local coordinates
        )

        # Prepare map writer
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        # Process map data
        if map_needs_writing:
            map_path = scene_path / "EXTRACTED" / "map.xodr"
            if map_path.exists():
                convert_xodr_map(map_path, map_writer)
            else:
                print(f"Warning: Map file {map_path} not found, skipping.")

        # Finalize map writing
        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""
        split, scene_path, batch_name = self._split_scene_path_pairs[log_index]

        # Load scene data
        extracted_path = scene_path / "EXTRACTED"
        datasource_summary = load_json(extracted_path / "datasource_summary.json")
        load_json(extracted_path / "data_info.json")

        # Optional files
        labels_path = scene_path / "labels.json"
        labels = load_json(labels_path) if labels_path.exists() else {}

        # Initialize log metadata
        log_metadata = LogMetadata(
            dataset="nurec",
            split=split,
            log_name=scene_path.name,
            location=scene_path.name,
            timestep_seconds=NUREC_TARGET_DT,
            vehicle_parameters=_get_nurec_vehicle_parameters(),
            camera_metadata=_get_nurec_camera_metadata(
                scene_path,
                datasource_summary,
                self.dataset_converter_config,
            ),
            lidar_metadata={},  # NUREC doesn't provide actual LiDAR data
            map_metadata=MapMetadata(
                dataset="nurec",
                split=split,
                log_name=scene_path.name,
                location=scene_path.name,
                map_has_z=True,
                map_is_local=True,
            ),
        )

        # Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            # Load rig trajectories from separate file
            rig_trajectories_data = load_json(extracted_path / "rig_trajectories.json")

            # Extract ego trajectory - it's in a list with one element
            rig_traj_entry = rig_trajectories_data["rig_trajectories"][0]
            rig_poses_list = rig_traj_entry["T_rig_worlds"]
            rig_timestamps_us = np.array(rig_traj_entry["T_rig_world_timestamps_us"], dtype=np.int64)

            # Convert trajectory matrices to numpy arrays
            rig_poses = [np.array(matrix, dtype=np.float64) for matrix in rig_poses_list]

            # Determine sampling timestamps (10Hz)
            start_timestamp = rig_timestamps_us[0]
            end_timestamp = rig_timestamps_us[-1]
            target_dt_us = int(NUREC_TARGET_DT * 1_000_000)  # Convert to microseconds

            sampling_timestamps = np.arange(start_timestamp, end_timestamp + target_dt_us, target_dt_us, dtype=np.int64)

            # Load track data
            sequence_tracks_path = extracted_path / "sequence_tracks.json"
            if sequence_tracks_path.exists():
                sequence_tracks = load_json(sequence_tracks_path)
                tracks_data = sequence_tracks.get("dummy_chunk_id", {}).get("tracks_data", {})
            else:
                tracks_data = {}

            # Load camera data if cameras are enabled
            camera_video_path = None
            camera_name = None
            camera_timestamps = None
            camera_extrinsic = None
            camera_type = None

            if self.dataset_converter_config.include_cameras:
                # Get available camera
                camera_result = get_available_camera(scene_path)
                if camera_result is not None:
                    camera_name, camera_video_path = camera_result
                    camera_type = NUREC_CAMERA_MAPPING.get(camera_name)

                    # Get camera timestamps from rig_trajectories
                    camera_timestamps_dict = rig_traj_entry.get("cameras_frame_timestamps_us", {})

                    # Find matching camera timestamps
                    for cam_key, timestamps in camera_timestamps_dict.items():
                        if camera_name in cam_key:
                            camera_timestamps = timestamps
                            break

                    # Get camera extrinsic from datasource_summary
                    camera_calibrations = datasource_summary.get("rig_trajectories", {}).get("camera_calibrations", {})
                    for cal_key, cal_data in camera_calibrations.items():
                        if cal_data["logical_sensor_name"] == camera_name:
                            # Convert T_sensor_rig matrix to StateSE3
                            T_sensor_rig = np.array(cal_data["T_sensor_rig"], dtype=np.float64)
                            camera_extrinsic = StateSE3.from_transformation_matrix(T_sensor_rig)
                            break

            # Process each sampling timestamp
            for timestamp_us in sampling_timestamps:
                # Interpolate ego pose
                ego_pose_matrix = interpolate_pose(timestamp_us, rig_timestamps_us, rig_poses)
                if ego_pose_matrix is None:
                    continue

                # Extract ego state
                ego_state = _extract_nurec_ego_state(ego_pose_matrix, log_metadata.vehicle_parameters)

                # Extract box detections
                box_detections = _extract_nurec_box_detections(
                    tracks_data,
                    timestamp_us,
                    ego_state,
                )

                # Extract scenario tags from labels
                scenario_tags = _extract_nurec_scenario_tags(labels)

                # Extract camera data
                cameras = _extract_nurec_cameras(
                    camera_video_path,
                    camera_timestamps,
                    camera_type,
                    camera_extrinsic,
                    timestamp_us,
                    self.dataset_converter_config,
                    self._nurec_data_root,
                )

                # Write log entry
                log_writer.write(
                    timestamp=TimePoint.from_us(int(timestamp_us)),
                    ego_state=ego_state,
                    box_detections=box_detections,
                    scenario_tags=scenario_tags,
                    cameras=cameras,
                    lidars={},  # No LiDAR data in NUREC
                )

        # Finalize log writing
        log_writer.close()


def _get_nurec_vehicle_parameters() -> VehicleParameters:
    """
    Get vehicle parameters for NUREC dataset.

    NUREC doesn't specify vehicle parameters, so we use generic values.
    """
    return VehicleParameters(
        vehicle_name="nurec_generic",
        width=2.0,
        length=4.5,
        height=1.5,
        wheel_base=2.7,
        rear_axle_to_center_vertical=0.75,  # Half of height
        rear_axle_to_center_longitudinal=1.35,  # Half of rear_overhang + wheel_base
    )


def _get_nurec_camera_metadata(
    scene_path: Path,
    datasource_summary: Dict,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraType, PinholeCameraMetadata]:
    """
    Extract camera metadata for NUREC dataset.

    Args:
        scene_path: Path to scene directory
        datasource_summary: Loaded datasource_summary.json
        dataset_converter_config: Converter configuration

    Returns:
        Dictionary mapping camera types to metadata
    """
    camera_metadata: Dict[PinholeCameraType, PinholeCameraMetadata] = {}

    if not dataset_converter_config.include_cameras:
        return camera_metadata

    # Find which camera video actually exists
    camera_result = get_available_camera(scene_path)
    if camera_result is None:
        return camera_metadata

    camera_name, video_path = camera_result

    # Get video info to validate
    video_info = get_video_info(video_path)
    if video_info is None:
        print(f"Warning: Could not read video info from {video_path}")
        return camera_metadata

    # Get camera calibration from datasource_summary
    # Find the calibration for this camera
    camera_calibrations = datasource_summary.get("rig_trajectories", {}).get("camera_calibrations", {})

    # Find matching calibration (keys have @sequence_id suffix)
    matching_cal = None
    for cal_key, cal_data in camera_calibrations.items():
        if cal_data["logical_sensor_name"] == camera_name:
            matching_cal = cal_data
            break

    if matching_cal is None:
        print(f"Warning: No calibration found for camera {camera_name}")
        return camera_metadata

    # Map NUREC camera to py123d camera type
    camera_type = NUREC_CAMERA_MAPPING.get(camera_name)
    if camera_type is None:
        print(f"Warning: Unknown camera name {camera_name}")
        return camera_metadata

    # Extract camera model parameters
    cam_model = matching_cal["camera_model"]
    params = cam_model["parameters"]

    # I think nurec uses "ftheta" fisheye model with polynomial distortion
    # Currently pinhole + radial distortion
    # For now, use simplified intrinsics and zero distortion
    # TODO: Implement proper ftheta to pinhole+distortion conversion

    width, height = params["resolution"]
    cx, cy = params["principal_point"]

    # The second coefficient is approximately the focal length ????
    angle_to_pixeldist_poly = params["angle_to_pixeldist_poly"]
    f_approx = angle_to_pixeldist_poly[1]  # Approximate focal length

    intrinsics = PinholeIntrinsics(
        fx=f_approx,
        fy=f_approx,
        cx=cx,
        cy=cy,
        skew=0.0,
    )

    # For now, use zero distortion (TODO: fit distortion model ??)
    distortion = PinholeDistortion(
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
    )

    camera_metadata[camera_type] = PinholeCameraMetadata(
        camera_type=camera_type,
        width=width,
        height=height,
        intrinsics=intrinsics,
        distortion=distortion,
    )

    return camera_metadata


def _extract_nurec_ego_state(
    pose_matrix: npt.NDArray[np.float64],
    vehicle_parameters: VehicleParameters,
) -> EgoStateSE3:
    """
    Extract ego state from NUREC pose matrix.

    Args:
        pose_matrix: 4x4 transformation matrix (T_rig_world)
        vehicle_parameters: Vehicle parameters

    Returns:
        EgoStateSE3 object
    """
    # Convert transformation matrix to StateSE3
    pose_7d = transformation_matrix_to_pose_7d(pose_matrix)
    center_se3 = StateSE3(
        x=pose_7d[0],
        y=pose_7d[1],
        z=pose_7d[2],
        qw=pose_7d[3],
        qx=pose_7d[4],
        qy=pose_7d[5],
        qz=pose_7d[6],
    )

    # TODO: Compute dynamic state from pose sequence (velocity, acceleration)?
    # For now, set to zero like in AV2 converter
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )

    return EgoStateSE3(
        center_se3=center_se3,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
        timepoint=None,  # Set during writing
    )


def _extract_nurec_box_detections(
    tracks_data: Dict,
    timestamp_us: int,
    ego_state: EgoStateSE3,
) -> BoxDetectionWrapper:
    """
    Extract box detections at a given timestamp from NUREC track data.

    Args:
        tracks_data: Track data from sequence_tracks.json
        timestamp_us: Query timestamp in microseconds
        ego_state: Current ego state for coordinate transformation

    Returns:
        BoxDetectionWrapper containing all detections at this timestamp
    """
    if not tracks_data:
        return BoxDetectionWrapper(box_detections=[])

    box_detections: List[BoxDetectionSE3] = []

    track_ids = tracks_data.get("tracks_id", [])
    track_poses = tracks_data.get("tracks_poses", [])
    track_timestamps = tracks_data.get("tracks_timestamps_us", [])
    track_classes = tracks_data.get("tracks_label_class", [])

    # Iterate through all tracks
    for track_idx in range(len(track_ids)):
        track_id = track_ids[track_idx]
        poses = track_poses[track_idx]
        timestamps = np.array(track_timestamps[track_idx], dtype=np.int64)
        label_class = track_classes[track_idx]

        # Find nearest timestamp for this track
        nearest_idx = find_nearest_timestamp_index(timestamps, timestamp_us, max_time_diff_us=100_000)  # 100ms

        if nearest_idx is None:
            continue

        # Get pose at this timestamp [x, y, z, qx, qy, qz, qw]
        pose = poses[nearest_idx]

        # Get default dimensions for this class
        dimensions = NUREC_DEFAULT_BOX_DIMENSIONS.get(label_class, (1.0, 1.0, 1.0))

        # Create bounding box
        bounding_box = BoundingBoxSE3(
            center=StateSE3(
                x=pose[0],
                y=pose[1],
                z=pose[2],
                qx=pose[3],
                qy=pose[4],
                qz=pose[5],
                qw=pose[6],
            ),
            length=dimensions[0],
            width=dimensions[1],
            height=dimensions[2],
        )

        # Map to detection type
        detection_type = NUREC_TO_DETECTION_TYPE.get(label_class, DetectionType.GENERIC_OBJECT)

        # Create detection
        detection = BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                detection_type=detection_type,
                timepoint=None,
                track_token=track_id,
                confidence=None,  # confidence not provided in nurec
            ),
            bounding_box_se3=bounding_box,
            velocity=Vector3D(x=0.0, y=0.0, z=0.0),  # TODO: Compute from track history
        )

        box_detections.append(detection)

    return BoxDetectionWrapper(box_detections=box_detections)


def _extract_nurec_scenario_tags(labels: Dict) -> List[str]:
    """
    Extract scenario tags from NUREC labels.json.

    Args:
        labels: Dictionary from labels.json

    Returns:
        List of scenario tag strings
    """
    tags = []

    # Flatten all label categories into tags
    for key, value in labels.items():
        if isinstance(value, list):
            tags.extend([f"{key}:{v}" for v in value])
        elif isinstance(value, bool):
            if value:
                tags.append(key)
        elif isinstance(value, str):
            tags.append(f"{key}:{value}")

    return tags if tags else ["unknown"]


def _extract_nurec_cameras(
    video_path: Optional[Path],
    camera_timestamps: Optional[List[List[int]]],
    camera_type: Optional[PinholeCameraType],
    camera_extrinsic: Optional[StateSE3],
    timestamp_us: int,
    dataset_converter_config: DatasetConverterConfig,
    nurec_data_root: Path,
) -> Dict[PinholeCameraType, Tuple[Union[str, bytes, npt.NDArray[np.uint8]], StateSE3]]:
    """
    Extract camera data for NUREC dataset at a given timestamp.

    Args:
        video_path: Path to camera video file
        camera_timestamps: List of [start, end] timestamp pairs for each frame
        camera_type: Camera type
        camera_extrinsic: Camera extrinsic transformation
        timestamp_us: Query timestamp in microseconds
        dataset_converter_config: Converter configuration
        nurec_data_root: Root directory for NUREC dataset

    Returns:
        Dictionary mapping camera type to (data, extrinsic) tuple
    """
    camera_dict: Dict[PinholeCameraType, Tuple[Union[str, bytes, npt.NDArray[np.uint8]], StateSE3]] = {}

    if not dataset_converter_config.include_cameras:
        return camera_dict

    # Check if we have all necessary data
    if video_path is None or camera_timestamps is None or camera_type is None or camera_extrinsic is None:
        return camera_dict

    # Find nearest frame
    frame_idx = find_nearest_camera_frame_index(camera_timestamps, timestamp_us, max_time_diff_us=100_000)

    if frame_idx is None:
        return camera_dict

    # Extract camera data based on storage option
    camera_data: Optional[Union[str, bytes, npt.NDArray[np.uint8]]] = None

    if dataset_converter_config.camera_store_option == "path":
        # Store video path with frame index (relative to nurec_data_root)
        # Format: "relative_video_path#frame_index" so it can be parsed later
        # TODO : Not sure how this should be handled later?
        relative_video_path = video_path.relative_to(nurec_data_root)
        camera_data = f"{relative_video_path}#{frame_idx}"

    elif dataset_converter_config.camera_store_option == "binary":
        # Extract frame and encode as JPEG
        frame = extract_video_frame(video_path, frame_idx)
        if frame is not None:
            import cv2

            # Encode as JPEG
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            camera_data = encoded.tobytes()

    if camera_data is not None:
        camera_dict[camera_type] = (camera_data, camera_extrinsic)

    return camera_dict
