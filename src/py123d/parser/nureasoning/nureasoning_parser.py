from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LogMetadata,
    MapMetadata,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.nureasoning.nureasoning_map_parser import NureasoningMapParser
from py123d.parser.nureasoning.utils.nureasoning_constants import (
    NUREASONING_BOX_DETECTIONS_SE3_METADATA,
    NUREASONING_CAMERA_ID_MAPPING,
    NUREASONING_CAMERA_KEY_MAPPING,
    NUREASONING_DATA_SPLITS,
    NUREASONING_DEFAULT_DT,
    NUREASONING_DETECTION_NAME_DICT,
    NUREASONING_LIDAR_DICT,
    NUREASONING_LIDAR_SWEEP_DURATION_US,
    NUREASONING_PARTS,
    NUREASONING_REAR_AXLE_HEIGHT,
    NUREASONING_TRAFFIC_STATUS_DICT,
)
from py123d.parser.nureasoning.utils.nureasoning_schema import Annotations, load_schema_pickle
from py123d.parser.registry import NureasoningBoxDetectionLabel

if TYPE_CHECKING:
    from py123d.parser.nureasoning.nureasoning_download import NureasoningDownloader

logger = logging.getLogger(__name__)


class NureasoningParser(BaseDatasetParser):
    """Dataset parser for the nuReasoning dataset."""

    def __init__(
        self,
        splits: List[str],
        nureasoning_data_root: Optional[Union[Path, str]] = None,
        log_names: Optional[List[str]] = None,
        downloader: Optional["NureasoningDownloader"] = None,
    ) -> None:
        """Initializes the :class:`NureasoningParser`.

        :param splits: Splits to convert. Available: ``"nureasoning-mini_train"``.
        :param nureasoning_data_root: Root of an already-extracted dataset
            (``<root>/<split>/<part>/<clip>/...``). Required when ``downloader`` is
            ``None``; ignored otherwise.
        :param log_names: Reserved for future per-log filtering (not yet implemented).
        :param downloader: Optional
            :class:`~py123d.parser.nureasoning.nureasoning_download.NureasoningDownloader`
            for streaming mode. When provided, the selected clips are materialized once
            into a session-scoped :class:`tempfile.TemporaryDirectory` (deleted when this
            parser is garbage-collected), and both log and map parsers read from it just
            like local mode. Clip selection (splits/parts/log_names/num_logs) is driven
            by the downloader; ``nureasoning_data_root`` is not required in this mode.
        """
        for split in splits:
            assert split in NUREASONING_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUREASONING_DATA_SPLITS}"
            )

        self._splits = splits
        self._log_names = log_names
        self._downloader = downloader
        # Handle for the streaming temp dir (kept alive for the parser's lifetime). Set early so
        # __del__ is safe even if materialization below raises.
        self._stream_temp_dir_handle: Optional[tempfile.TemporaryDirectory] = None

        if downloader is not None:
            self._nureasoning_data_root = self._materialize_streaming_root(downloader)
        else:
            assert nureasoning_data_root is not None, (
                "`nureasoning_data_root` must be provided when `downloader` is None."
            )
            self._nureasoning_data_root = Path(nureasoning_data_root)

        self._split_log_path_pairs: List[Tuple[str, Path]] = self._collect_split_log_path_pairs()

    def _materialize_streaming_root(self, downloader: "NureasoningDownloader") -> Path:
        """Download the selected clips into a managed temp dir and return it as the data root.

        Mirrors the nuScenes streaming model: because nuReasoning maps are per-log
        (``map.pkl`` lives inside each clip), the cleanest way to feed both the log and
        map parsers is to materialize the selected subset once into a temp directory that
        mirrors the on-disk layout, then read from it exactly like local mode. The temp
        dir (and the extracted clips) is removed in :meth:`__del__`.
        """
        self._stream_temp_dir_handle = tempfile.TemporaryDirectory(prefix="py123d-nureasoning-")
        tmp_root = Path(self._stream_temp_dir_handle.name)
        # The BaseDownloader contract lets the parser assign output_dir when it is None.
        downloader.output_dir = tmp_root
        logger.info("nuReasoning streaming: materializing selected clips into %s", tmp_root)
        downloader.download()
        return tmp_root

    def __del__(self) -> None:
        handle = getattr(self, "_stream_temp_dir_handle", None)
        if handle is not None:
            handle.cleanup()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, Path]]:
        """Collects the (split, log_path) pairs for the specified splits."""
        split_log_path_pairs: List[Tuple[str, Path]] = []

        for split in self._splits:
            split_type = split.split("_")[-1]
            assert split_type in {"train", "val", "test"}

            if split in {"nureasoning-mini_train"}:
                nureasoning_split_folder = self._nureasoning_data_root / "train"
            else:
                raise NotImplementedError(f"nuReasoning split {split} is not yet available.")

            valid_log_folders: List[Path] = []
            for part_folder in sorted(nureasoning_split_folder.iterdir()):
                if part_folder.is_dir() and part_folder.name in NUREASONING_PARTS[split]:
                    for log_folder in sorted(part_folder.iterdir()):
                        if log_folder.is_dir():
                            valid_log_folders.append(log_folder)

            if self._log_names is not None:
                raise NotImplementedError("Filtering by log names is not yet implemented for the nuReasoning parser.")

            for log_folder in valid_log_folders:
                split_log_path_pairs.append((split, log_folder))

        return split_log_path_pairs

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        # nuReasoning maps are per-log: one ``map.pkl`` per clip, so one map parser per log.
        return [
            NureasoningMapParser(
                split=split,
                log_name=source_log_path.name,
                source_log_path=source_log_path,
            )
            for split, source_log_path in self._split_log_path_pairs
        ]

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            NureasoningLogParser(
                split=split,
                source_log_path=source_log_path,
                nureasoning_data_root=self._nureasoning_data_root,
            )
            for split, source_log_path in self._split_log_path_pairs
        ]


class NureasoningLogParser(BaseLogParser):
    """Lightweight, picklable handle to one nuReasoning log."""

    def __init__(
        self,
        split: str,
        source_log_path: Path,
        nureasoning_data_root: Path,
    ) -> None:
        self._split = split
        self._source_log_path = source_log_path
        self._nureasoning_data_root = nureasoning_data_root

    def _get_nureasoning_metadata_json(self) -> Dict[str, Any]:
        """Helper function to load the nuReasoning metadata JSON for this log."""
        metadata_json_path = self._source_log_path / "metadata.json"
        if not metadata_json_path.exists() or not metadata_json_path.is_file():
            raise FileNotFoundError(
                f"Metadata JSON file not found for log {self._source_log_path}: {metadata_json_path}"
            )

        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)

        return metadata_json

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        metadata_json = self._get_nureasoning_metadata_json()
        # NOTE: Use the full folder name (``<log_name>_<clip_token>``) as the unique log id. The
        # folder name contains dots, so ``Path.stem`` would truncate it; ``metadata["log_name"]``
        # omits the clip token and is not guaranteed unique across clips.
        log_name = self._source_log_path.name
        location = metadata_json.get("clip_location", None)
        location = location.replace(" ", "-") if location else None

        # Each clip ships a per-log ``map.pkl`` (converted by NureasoningMapParser). The map is stored in 2D
        # (all map geometry has z == 0), and routed by (split, log_name) at read time.
        map_metadata = MapMetadata(
            dataset="nureasoning",
            split=self._split,
            log_name=log_name,
            location=location,
            map_has_z=False,
            map_is_per_log=True,
        )
        return LogMetadata(
            dataset="nureasoning",
            split=self._split,
            log_name=log_name,
            location=location,
            map_metadata=map_metadata,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        metadata_json = self._get_nureasoning_metadata_json()

        ego_state_se3_metadata = _get_nureasoning_ego_state_se3_metadata(metadata_json)
        camera_metadatas = _get_nureasoning_camera_metadata(self._source_log_path, metadata_json)
        lidar_merged_metadata = _get_nureasoning_lidar_merged_metadata()
        box_detections_se3_metadata = NUREASONING_BOX_DETECTIONS_SE3_METADATA
        scenario_type = metadata_json.get("scenario_type", None)

        for frame in metadata_json["frames"]:
            timestamp = Timestamp.from_us(frame["timestamp_us"])

            # 1. Ego State
            ego_state_se3 = _extract_nureasoning_ego_state(self._source_log_path, frame, ego_state_se3_metadata)
            ego_trajectory = _extract_nureasoning_ego_trajectory(self._source_log_path, frame, timestamp)

            # 2. Annotations
            annotations = load_schema_pickle(self._source_log_path / frame["annotations"])
            assert isinstance(annotations, Annotations), f"Expected Annotations object, got {type(annotations)}"
            box_detections_se3 = _extract_nureasoning_box_detections(
                annotations, timestamp, box_detections_se3_metadata
            )
            traffic_lights = _extract_nureasoning_traffic_lights(annotations, timestamp)

            # 3. Sensors
            parsed_cameras = _extract_nureasoning_cameras(
                source_log_path=self._source_log_path,
                nureasoning_data_root=self._nureasoning_data_root,
                frame=frame,
                ego_state_se3=ego_state_se3,
                camera_metadatas=camera_metadatas,
            )

            # The box detections are the sync reference column, so we always emit them (even empty).
            modalities: List[BaseModality] = [ego_state_se3, box_detections_se3, traffic_lights, ego_trajectory]
            modalities.extend(parsed_cameras)

            parsed_lidar = _extract_nureasoning_lidar_data(
                self._source_log_path, self._nureasoning_data_root, frame, lidar_merged_metadata
            )
            if parsed_lidar is not None:
                modalities.append(parsed_lidar)

            reasoning = _extract_nureasoning_reasoning(self._source_log_path, frame, timestamp)
            if reasoning is not None:
                modalities.append(reasoning)

            scenario = _extract_nureasoning_scenario(frame, scenario_type, timestamp)
            if scenario is not None:
                modalities.append(scenario)

            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_nureasoning_ego_state_se3_metadata(metadata_json: Dict[str, Any]) -> EgoStateSE3Metadata:
    """Extracts the nuReasoning ego state SE3 metadata for a given log."""
    # NOTE @DanielDauner: Assuming Hyundai Ioniq 5 vehicle model.
    # https://en.wikipedia.org/wiki/Hyundai_Ioniq_5
    ego_dimensions = metadata_json.get("ego_dimensions", None)
    assert ego_dimensions is not None, "Ego dimensions not found in metadata JSON."

    _length = ego_dimensions["length"]
    _height = ego_dimensions["height"]

    # NOTE: Assuming distance from rear-axle to vehicle rear. Needs verification (see TODO.md).
    vehicle_rear_length = ego_dimensions["vehicle_rear_length"]

    # TODO @DanielDauner: Verify these values, specifically once lidar available.
    half_length = _length / 2.0
    rear_axle_to_center_longitudinal = half_length - vehicle_rear_length
    rear_axle_to_center_vertical = (_height / 2.0) - NUREASONING_REAR_AXLE_HEIGHT

    center_to_imu_se3 = PoseSE3.from_R_t(
        rotation=np.zeros((3,), dtype=np.float64),
        translation=np.array([rear_axle_to_center_longitudinal, 0.0, rear_axle_to_center_vertical], dtype=np.float64),
    )

    return EgoStateSE3Metadata(
        vehicle_name="TODO",  # TODO @DanielDauner: Set proper vehicle name.
        width=ego_dimensions["width"],
        length=ego_dimensions["length"],
        height=ego_dimensions["height"],
        wheel_base=3.000,  # [m] NOTE @DanielDauner: Value from Wikipedia, needs verification.
        center_to_imu_se3=center_to_imu_se3,
        # NOTE: Assuming rear axle and IMU are co-located. Should be verified for nuReasoning (see TODO.md).
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _get_nureasoning_camera_metadata(
    source_log_path: Path, metadata_json: Dict[str, Any]
) -> Dict[CameraID, PinholeCameraMetadata]:
    """Extracts the nuReasoning camera metadata for a given log."""
    camera_metadatas: Dict[CameraID, PinholeCameraMetadata] = {}
    camera_calibrations_json: Dict[str, Any] = metadata_json.get("camera_calibrations", {})

    for camera_id, camera_name in NUREASONING_CAMERA_ID_MAPPING.items():
        has_valid_path = (source_log_path / "cameras" / camera_name).exists()
        has_metadata = camera_name in camera_calibrations_json.keys()

        if has_valid_path and has_metadata:
            camera_calibration_json = camera_calibrations_json[camera_name]

            _width, _height = camera_calibration_json["width"], camera_calibration_json["height"]

            # NOTE: The extrinsic is camera->lidar. We treat it as camera->IMU, which is correct only if
            # the lidar, IMU, and ego-pose origin coincide. Needs verification (see TODO.md).
            extrinsic = PoseSE3.from_R_t(
                rotation=np.array(camera_calibration_json["sensor2lidar_rotation"], dtype=np.float64),
                translation=np.array(camera_calibration_json["sensor2lidar_translation"], dtype=np.float64),
            )

            camera_matrix = np.array(camera_calibration_json["intrinsic"], dtype=np.float64).reshape(3, 3)
            intrinsic = PinholeIntrinsics.from_camera_matrix(camera_matrix)

            camera_metadatas[camera_id] = PinholeCameraMetadata(
                camera_name=camera_name,
                camera_id=camera_id,
                width=_width,
                height=_height,
                intrinsics=intrinsic,
                distortion=None,  # TODO @DanielDauner: Verify if images are rectified (see TODO.md).
                camera_to_imu_se3=extrinsic,
                is_undistorted=True,  # TODO @DanielDauner: Verify if correct.
            )

    return camera_metadatas


# ------------------------------------------------------------------------------------------------------------------
# Modality extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nureasoning_ego_state(
    source_log_path: Path,
    frame: Dict[str, Any],
    ego_state_se3_metadata: EgoStateSE3Metadata,
    # timestamp: Timestamp,
) -> EgoStateSE3:
    """Extracts the nuReasoning ego state from the per-frame ego_state pickle."""

    file_string = frame["ego_state"]
    timestamp = Timestamp.from_us(int(file_string.split("/")[-1].removesuffix(".pkl")))
    ego_state = load_schema_pickle(source_log_path / file_string)
    pose = ego_state.pose
    velocity = ego_state.velocity
    acceleration = ego_state.acceleration

    imu_pose = PoseSE3(
        x=pose["x"], y=pose["y"], z=pose["z"], qw=pose["qw"], qx=pose["qx"], qy=pose["qy"], qz=pose["qz"]
    )
    dynamic_state_se3 = DynamicStateSE3(
        velocity=Vector3D(x=velocity["vx"], y=velocity["vy"], z=velocity["vz"]),
        acceleration=Vector3D(x=acceleration["ax"], y=acceleration["ay"], z=acceleration["az"]),
        # NOTE: Angular velocity is not provided by nuReasoning.
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        metadata=ego_state_se3_metadata,
        dynamic_state_se3=dynamic_state_se3,
        timestamp=timestamp,
    )


def _extract_nureasoning_ego_trajectory(
    source_log_path: Path, frame: Dict[str, Any], timestamp: Timestamp
) -> CustomModality:
    """Extracts the ego history/future trajectory from the per-frame ego_state pickle as a custom modality.

    Both polylines are ``[x, y, yaw]`` samples in the global frame at the dataset frame rate (history grows
    up to 3 s, future up to 5 s). Either may be empty at the log boundaries, in which case it is stored as a
    ``(0, 3)`` array. The trajectory is in the same global frame as ``EgoStateSE3.imu_se3``.
    """
    ego_state = load_schema_pickle(source_log_path / frame["ego_state"])
    history_global = np.asarray(ego_state.trajectory_history or [], dtype=np.float64).reshape(-1, 3)
    future_global = np.asarray(ego_state.trajectory_future or [], dtype=np.float64).reshape(-1, 3)

    return CustomModality(
        data={"history_global": history_global, "future_global": future_global},
        metadata=CustomModalityMetadata(
            modality_id="ego_trajectory",
            metadata={"frame": "global", "columns": ["x", "y", "yaw"], "dt_s": NUREASONING_DEFAULT_DT},
        ),
        timestamp=timestamp,
    )


def _extract_nureasoning_box_detections(
    annotations: Annotations, timestamp: Timestamp, box_detections_se3_metadata: BoxDetectionsSE3Metadata
) -> BoxDetectionsSE3:
    """Extracts the nuReasoning box detections from the per-frame annotations pickle."""

    box_detections: List[BoxDetectionSE3] = []
    for obj in annotations.objects:
        pose, velocity, dimensions = obj.pose, obj.velocity, obj.dimensions
        quaternion = EulerAngles(roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=pose["yaw"]).quaternion
        bounding_box = BoundingBoxSE3(
            center_se3=PoseSE3(
                x=pose["x"],
                y=pose["y"],
                z=pose["z"],
                qw=quaternion.qw,
                qx=quaternion.qx,
                qy=quaternion.qy,
                qz=quaternion.qz,
            ),
            length=dimensions["l"],
            width=dimensions["w"],
            height=dimensions["h"],
        )
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    # NOTE: Fall back to OTHER_OTHER for categories not in the mapping.
                    label=NUREASONING_DETECTION_NAME_DICT.get(obj.category, NureasoningBoxDetectionLabel.OTHER_OTHER),
                    track_token=obj.track_token,
                ),
                bounding_box_se3=bounding_box,
                # NOTE: We assume object velocity is in the global frame, as expected by
                # BoxDetectionSE3.velocity_3d. Needs verification.
                velocity_3d=Vector3D(x=velocity["vx"], y=velocity["vy"], z=velocity["vz"]),
            )
        )

    return BoxDetectionsSE3(
        box_detections=box_detections,
        timestamp=timestamp,
        metadata=box_detections_se3_metadata,
    )


def _extract_nureasoning_traffic_lights(annotations: Annotations, timestamp: Timestamp) -> TrafficLightDetections:
    """Extracts the nuReasoning traffic light detections from the per-frame annotations pickle."""

    detections: List[TrafficLightDetection] = []
    for traffic_light in annotations.traffic_light_states:
        # Prefer the lane connector as the lane reference; fall back to the roadblock id.
        # Skip if neither is available, since we cannot anchor it without a map.
        lane_id = traffic_light.lane_connector_id
        if lane_id is None:
            lane_id = traffic_light.roadblock_id

        if lane_id is not None:
            detections.append(
                TrafficLightDetection(
                    lane_id=int(lane_id),
                    status=NUREASONING_TRAFFIC_STATUS_DICT[traffic_light.state],
                )
            )

    return TrafficLightDetections(detections=detections, timestamp=timestamp)


def _extract_nureasoning_cameras(
    source_log_path: Path,
    nureasoning_data_root: Path,
    frame: Dict[str, Any],
    ego_state_se3: EgoStateSE3,
    camera_metadatas: Dict[CameraID, PinholeCameraMetadata],
) -> List[ParsedCamera]:
    """Extracts the nuReasoning camera data for all cameras available in this frame.

    The camera-to-global pose is composed from the ego (IMU) pose and the static camera-to-IMU
    extrinsic. Image bytes are not loaded here; the log writer reads them at write time.
    """
    camera_paths: Dict[str, str] = frame.get("sensors", {}).get("cameras", {})
    camera_data_list: List[ParsedCamera] = []

    for camera_id, camera_metadata in camera_metadatas.items():
        frame_key = NUREASONING_CAMERA_KEY_MAPPING[camera_id]
        relative_camera_path = camera_paths.get(frame_key, None)
        if relative_camera_path is None:
            continue

        full_image_path = source_log_path / relative_camera_path
        if not (full_image_path.exists() and full_image_path.is_file()):
            continue

        camera_to_global_se3 = rel_to_abs_se3(
            origin=ego_state_se3.imu_se3,
            pose_se3=camera_metadata.camera_to_imu_se3,
        )

        timestamp = Timestamp.from_us(int(relative_camera_path.split("/")[-1].removesuffix(".jpg").split("_")[-1]))

        camera_data_list.append(
            ParsedCamera(
                metadata=camera_metadata,
                timestamp=timestamp,
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=nureasoning_data_root,
                relative_path=full_image_path.relative_to(nureasoning_data_root),
            )
        )

    return camera_data_list


def _get_nureasoning_lidar_merged_metadata() -> LidarMergedMetadata:
    """Builds the merged-lidar metadata for nuReasoning.

    The point cloud merges multiple lidar sensors (see ``NUREASONING_LIDAR_DICT``). The points are
    already in the common ego/lidar frame, so per-sensor extrinsics are the identity.
    """
    # NOTE: lidar == IMU is assumed, so the extrinsics are the identity (see TODO.md).
    metadata: Dict[LidarID, LidarMetadata] = {
        lidar_id: LidarMetadata(
            lidar_name=lidar_id.serialize(),
            lidar_id=lidar_id,
            lidar_to_imu_se3=PoseSE3.identity(),
        )
        for lidar_id in NUREASONING_LIDAR_DICT.values()
    }
    return LidarMergedMetadata(metadata)


def _extract_nureasoning_lidar_data(
    source_log_path: Path,
    nureasoning_data_root: Path,
    frame: Dict[str, Any],
    lidar_merged_metadata: LidarMergedMetadata,
) -> Optional[ParsedLidar]:
    """Extracts the nuReasoning lidar data from the per-frame lidar path, if present.

    Only the path is stored (see the ``lidar_store_option: "path"`` conversion config); the point
    cloud is decoded at read time by ``nureasoning_sensor_io``. Lidar is only present in some logs
    (e.g. part_3).
    """
    parsed_lidar: Optional[ParsedLidar] = None

    relative_lidar_path = frame.get("sensors", {}).get("lidar", {}).get("point_cloud_path", None)
    if relative_lidar_path:
        full_lidar_path = source_log_path / relative_lidar_path

        if full_lidar_path.exists() and full_lidar_path.is_file():
            timestamp = Timestamp.from_us(int(relative_lidar_path.split("/")[-1].removesuffix(".pcd").split("_")[-1]))
            parsed_lidar = ParsedLidar(
                metadata=lidar_merged_metadata,
                start_timestamp=timestamp,
                end_timestamp=Timestamp.from_us(timestamp.time_us + NUREASONING_LIDAR_SWEEP_DURATION_US),
                dataset_root=nureasoning_data_root,
                relative_path=full_lidar_path.relative_to(nureasoning_data_root),
            )
        else:
            logger.debug(f"Lidar file not found: {full_lidar_path}")

    return parsed_lidar


def _extract_nureasoning_reasoning(
    source_log_path: Path, frame: Dict[str, Any], timestamp: Timestamp
) -> Optional[CustomModality]:
    """Extracts the nuReasoning reasoning annotations (raw passthrough) when present for this frame."""
    custom_modality: Optional[CustomModality] = None

    relative_reasoning_path = frame.get("reasoning", "")
    if relative_reasoning_path:
        with open(source_log_path / relative_reasoning_path, "r", encoding="utf-8") as f:
            reasoning_json = json.load(f)

        # CustomModality expects a dict with string keys; wrap non-dict payloads.
        data = reasoning_json if isinstance(reasoning_json, dict) else {"reasoning": reasoning_json}
        custom_modality = CustomModality(
            data=data,
            metadata=CustomModalityMetadata(modality_id="reasoning"),
            timestamp=timestamp,
        )

    return custom_modality


def _extract_nureasoning_scenario(
    frame: Dict[str, Any], scenario_type: Optional[str], timestamp: Timestamp
) -> Optional[CustomModality]:
    """Extracts the nuReasoning mission-goal / scenario metadata as a custom modality."""
    custom_modality: Optional[CustomModality] = None

    mission_goal = frame.get("mission_goal", None)
    if mission_goal is not None:
        data: Dict[str, Any] = {
            "command": mission_goal.get("command", None),
            "route_path": mission_goal.get("route_path", []),
            "scenario_type": scenario_type,
        }
        custom_modality = CustomModality(
            data=data,
            metadata=CustomModalityMetadata(modality_id="scenario"),
            timestamp=timestamp,
        )

    return custom_modality
