"""Vehicle-side parser for the Griffin aerial-ground cooperative dataset.

This parser converts the **ground-vehicle** agent of Griffin: four cardinal
pinhole cameras, the 80-beam top LiDAR, ego poses, and 3D box annotations. Each
Griffin scene (~150 frames at 10 Hz) becomes one 123D log, routed to its
official ``train`` / ``val`` partition via the embedded split files.

Scope note
----------
Griffin's distinguishing feature is aerial-ground cooperation (a drone agent
with its own sensors and annotations). The 123D log abstraction models a single
agent per log, so cooperative/drone-side data is deliberately left to a
follow-up: the vehicle-side conversion below is self-contained, matches the
shape of existing single-agent datasets (e.g. PandaSet), and provides the
foundation a later drone-side / cooperative parser can build on.

References
----------
.. [1] Wang et al., "Griffin: Aerial-Ground Cooperative Detection and Tracking
   Dataset and Benchmark", arXiv preprint 2503.06983 (2025).
.. [2] Official toolkit: https://github.com/wang-jh18-SVM/Griffin
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.geometry import BoundingBoxSE3, PoseSE3
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.griffin.utils.griffin_constants import (
    GRIFFIN_BOX_DETECTION_FROM_STR,
    GRIFFIN_BOX_DETECTIONS_SE3_METADATA,
    GRIFFIN_CAMERA_HEIGHT,
    GRIFFIN_CAMERA_WIDTH,
    GRIFFIN_LIDAR_PERIOD_US,
    GRIFFIN_LIDAR_SENSOR_NAME,
    GRIFFIN_SPLITS,
    GRIFFIN_VEHICLE_CAMERA_MAPPING,
    build_griffin_ego_metadata,
    split_to_subset_and_kind,
)
from py123d.parser.griffin.utils.griffin_utils import (
    ego_box_to_global_se3,
    load_calibration,
    load_scene_index,
    load_split_scene_names,
    parse_label_file,
    pose_dict_to_ego_to_global_se3,
    read_json,
    sensor_to_ego_pose_se3,
)

logger = logging.getLogger(__name__)

# Zero distortion: Griffin renders ideal pinhole cameras in CARLA.
_GRIFFIN_PINHOLE_DISTORTION = PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)


def _default_split_file(subset: str) -> Path:
    """Resolve the embedded official split file for ``subset``.

    :param subset: Subset name, e.g. ``"griffin_50scenes_25m"``.
    :return: Path to the packaged ``splits/{subset}.json``.
    """
    return Path(str(resources.files("py123d.parser.griffin").joinpath("splits", f"{subset}.json")))


class GriffinParser(BaseDatasetParser):
    """Dataset parser for the Griffin dataset (vehicle-side, single agent)."""

    def __init__(
        self,
        splits: List[str],
        griffin_data_root: Union[Path, str],
        split_data_root: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initialize the :class:`GriffinParser`.

        :param splits: Splits to convert, e.g. ``["griffin_50scenes_25m_train"]``.
            Each split's subset prefix selects the ``griffin-release`` tree and
            the official train/val partition.
        :param griffin_data_root: Directory containing the subset folders, laid
            out as ``{griffin_data_root}/{subset}/griffin-release/vehicle-side``
            (the official ``datasets`` directory).
        :param split_data_root: Optional directory holding ``{subset}.json`` split
            files. When ``None``, the official split files embedded in this
            package are used.
        """
        for split in splits:
            assert split in GRIFFIN_SPLITS, f"Split {split} is not available. Available splits: {GRIFFIN_SPLITS}"

        self._splits: List[str] = list(splits)
        self._griffin_data_root: Path = Path(griffin_data_root)
        self._split_data_root: Optional[Path] = Path(split_data_root) if split_data_root is not None else None

    def _split_file(self, subset: str) -> Path:
        """Return the split file for ``subset`` (override directory or embedded)."""
        if self._split_data_root is not None:
            return self._split_data_root / f"{subset}.json"
        return _default_split_file(subset)

    def _vehicle_root(self, subset: str) -> Path:
        """Return the ``vehicle-side`` directory for ``subset``."""
        return self._griffin_data_root / subset / "griffin-release" / "vehicle-side"

    def get_log_parsers(self) -> List[GriffinLogParser]:  # type: ignore[override]
        """Inherited, see superclass. One log per scene, routed to its split partition."""
        log_parsers: List[GriffinLogParser] = []
        for split in self._splits:
            subset, kind = split_to_subset_and_kind(split)
            vehicle_root = self._vehicle_root(subset)
            if not vehicle_root.is_dir():
                logger.warning("Griffin subset '%s' not found at %s; skipping split '%s'.", subset, vehicle_root, split)
                continue

            scene_index = load_scene_index(vehicle_root)
            wanted_scenes = set(load_split_scene_names(self._split_file(subset), kind))
            scene_frames = {name: frames for name, frames in scene_index}

            missing = wanted_scenes - set(scene_frames)
            if missing:
                logger.warning(
                    "Split '%s' lists %d scene(s) absent from %s/scene_infos.json (e.g. %s).",
                    split,
                    len(missing),
                    vehicle_root,
                    sorted(missing)[0],
                )

            for scene_name, frames in scene_index:
                if scene_name not in wanted_scenes or not frames:
                    continue
                log_parsers.append(
                    GriffinLogParser(
                        griffin_data_root=self._griffin_data_root,
                        subset=subset,
                        scene_name=scene_name,
                        frames=frames,
                        split=split,
                    )
                )
        return log_parsers

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass. Griffin provides no HD map."""
        return []


class GriffinLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Griffin scene (vehicle-side)."""

    def __init__(
        self,
        griffin_data_root: Path,
        subset: str,
        scene_name: str,
        frames: List[str],
        split: str,
    ) -> None:
        """Initialize the log parser.

        :param griffin_data_root: Directory containing the subset folders.
        :param subset: Subset name, e.g. ``"griffin_50scenes_25m"``.
        :param scene_name: Full scene identifier, e.g. ``scene-0026-Town07-001``.
        :param frames: Ordered 6-digit frame ids belonging to this scene.
        :param split: Owning split name.
        """
        self._griffin_data_root = griffin_data_root
        self._subset = subset
        self._scene_name = scene_name
        self._frames = frames
        self._split = split

        # ``{subset}/griffin-release`` is the sensor-root-relative prefix used for
        # all stored LiDAR/camera relative paths, so they resolve from a single
        # ``get_sensor_root("griffin") == griffin_data_root`` at read time.
        self._release_rel = Path(subset) / "griffin-release"
        self._vehicle_root = griffin_data_root / self._release_rel / "vehicle-side"
        self._calib_dir = self._vehicle_root / "calib"

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="griffin",
            split=self._split,
            log_name=self._scene_name,
            location=self._subset,
        )

    def _build_camera_metadata(self) -> Dict[CameraID, PinholeCameraMetadata]:
        """Build pinhole camera metadata from the static per-camera calibration files."""
        camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
        for camera_name, camera_id in GRIFFIN_VEHICLE_CAMERA_MAPPING.items():
            intrinsic, extrinsic = load_calibration(self._calib_dir, camera_name)
            assert intrinsic is not None, f"Missing intrinsics for Griffin camera '{camera_name}'."
            camera_metadata[camera_id] = PinholeCameraMetadata(
                camera_name=camera_name,
                camera_id=camera_id,
                intrinsics=PinholeIntrinsics(
                    fx=float(intrinsic[0, 0]),
                    fy=float(intrinsic[1, 1]),
                    cx=float(intrinsic[0, 2]),
                    cy=float(intrinsic[1, 2]),
                ),
                distortion=_GRIFFIN_PINHOLE_DISTORTION,
                width=GRIFFIN_CAMERA_WIDTH,
                height=GRIFFIN_CAMERA_HEIGHT,
                # Griffin's extrinsic is sensor-to-ego; the ego frame is the IMU
                # frame in 123D, so it maps directly to camera_to_imu_se3.
                camera_to_imu_se3=PoseSE3.from_transformation_matrix(extrinsic),
                is_undistorted=True,
            )
        return camera_metadata

    def _build_lidar_merged_metadata(self) -> LidarMergedMetadata:
        """Build the merged-LiDAR metadata for the single ego-frame top LiDAR."""
        return LidarMergedMetadata(
            {
                LidarID.LIDAR_TOP: LidarMetadata(
                    lidar_name=GRIFFIN_LIDAR_SENSOR_NAME,
                    lidar_id=LidarID.LIDAR_TOP,
                    lidar_to_imu_se3=sensor_to_ego_pose_se3(self._calib_dir, GRIFFIN_LIDAR_SENSOR_NAME),
                )
            }
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_metadata = build_griffin_ego_metadata()
        camera_metadata = self._build_camera_metadata()
        lidar_merged_metadata = self._build_lidar_merged_metadata()

        for frame in self._frames:
            timestamp = Timestamp.from_us(int(frame) * GRIFFIN_LIDAR_PERIOD_US)

            ego_pose = read_json(self._vehicle_root / "pose" / f"{frame}.json")
            ego_to_global = pose_dict_to_ego_to_global_se3(ego_pose)
            ego_state = EgoStateSE3.from_imu(
                imu_se3=ego_to_global,
                metadata=ego_metadata,
                timestamp=timestamp,
                dynamic_state_se3=None,
            )

            box_detections = self._extract_box_detections(frame, timestamp, ego_to_global)
            parsed_lidar = self._extract_lidar(frame, timestamp, lidar_merged_metadata)
            parsed_cameras = self._extract_cameras(frame, timestamp, ego_to_global, camera_metadata)

            yield ModalitiesSync(
                timestamp=timestamp,
                modalities=[ego_state, box_detections, parsed_lidar, *parsed_cameras],
            )

    def _extract_box_detections(self, frame: str, timestamp: Timestamp, ego_to_global: PoseSE3) -> BoxDetectionsSE3:
        """Extract 3D box detections for a frame, lifted from ego to global frame."""
        annotations = parse_label_file(self._vehicle_root / "label" / f"{frame}.txt")

        box_detections: List[BoxDetectionSE3] = []
        for annotation in annotations:
            label = GRIFFIN_BOX_DETECTION_FROM_STR.get(annotation["type"].lower())
            if label is None:
                # Non-traffic categories (e.g. CARLA military props) are outside
                # the Griffin perception taxonomy; skip rather than mislabel.
                logger.debug("Skipping unrecognized Griffin category '%s'.", annotation["type"])
                continue

            bounding_box = BoundingBoxSE3(
                center_se3=ego_box_to_global_se3(annotation, ego_to_global),
                length=annotation["l"],
                width=annotation["w"],
                height=annotation["h"],
            )
            box_detections.append(
                BoxDetectionSE3(
                    attributes=BoxDetectionAttributes(label=label, track_token=annotation["track_id"]),
                    bounding_box_se3=bounding_box,
                    velocity_3d=None,
                )
            )

        return BoxDetectionsSE3(
            box_detections=box_detections,
            timestamp=timestamp,
            metadata=GRIFFIN_BOX_DETECTIONS_SE3_METADATA,
        )

    def _extract_lidar(
        self, frame: str, timestamp: Timestamp, lidar_merged_metadata: LidarMergedMetadata
    ) -> ParsedLidar:
        """Reference the ego-frame ``.ply`` LiDAR sweep for a frame."""
        relative_path = self._release_rel / "vehicle-side" / "lidar" / "lidar_top" / f"{frame}.ply"
        absolute_path = self._griffin_data_root / relative_path
        assert absolute_path.exists(), f"Griffin LiDAR file does not exist: {absolute_path}"
        return ParsedLidar(
            metadata=lidar_merged_metadata,
            start_timestamp=timestamp,
            end_timestamp=Timestamp.from_us(timestamp.time_us + GRIFFIN_LIDAR_PERIOD_US),
            dataset_root=self._griffin_data_root,
            relative_path=str(relative_path),
        )

    def _extract_cameras(
        self,
        frame: str,
        timestamp: Timestamp,
        ego_to_global: PoseSE3,
        camera_metadata: Dict[CameraID, PinholeCameraMetadata],
    ) -> List[ParsedCamera]:
        """Reference all available camera images for a frame, with global camera poses."""
        parsed_cameras: List[ParsedCamera] = []
        for camera_name, camera_id in GRIFFIN_VEHICLE_CAMERA_MAPPING.items():
            relative_path = self._release_rel / "vehicle-side" / "camera" / camera_name / f"{frame}.png"
            absolute_path = self._griffin_data_root / relative_path
            if not absolute_path.exists():
                continue
            camera_to_global_se3 = rel_to_abs_se3(
                origin=ego_to_global,
                pose_se3=camera_metadata[camera_id].camera_to_imu_se3,
            )
            parsed_cameras.append(
                ParsedCamera(
                    metadata=camera_metadata[camera_id],
                    timestamp=timestamp,
                    camera_to_global_se3=camera_to_global_se3,
                    dataset_root=self._griffin_data_root,
                    relative_path=str(relative_path),
                )
            )
        return parsed_cameras
