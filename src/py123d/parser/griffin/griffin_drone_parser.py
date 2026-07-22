"""Drone-side parser for the Griffin aerial-ground cooperative dataset.

This parser converts the **UAV (drone)** agent of Griffin as a *second
single-agent log* (Path 1): five pinhole cameras (four cardinal + one nadir),
ego poses, and 3D box annotations. It deliberately mirrors the shape of the
vehicle-side :class:`~py123d.parser.griffin.griffin_parser.GriffinParser`, with
three agent-specific differences:

1. **No LiDAR.** Griffin equips only the ground vehicle with a LiDAR; the drone
   is camera-only (``use_lidar=False`` in the official drone-side configs).
2. **A nadir camera.** The drone adds a downward-looking ``bottom`` camera
   (mapped to :class:`~py123d.datatypes.sensors.CameraID.PCAM_D0`) on top of the
   four cardinal cameras.
3. **Aerial state in a custom modality.** "This ego is a UAV" plus the per-frame
   altitude are emitted via :class:`~py123d.datatypes.custom.CustomModality`
   rather than overloading the core ego/sensor types.

Because boxes and cameras are lifted to a shared global (ENU) world frame for
both agents, the vehicle and drone logs of the same scene land in one coordinate
system automatically; a downstream user aligns them by scene id + timestamp.
True first-class cooperation (cross-agent transforms as a core datatype) is a
separate, design-first follow-up (Path 2) and is intentionally out of scope.

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
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata
from py123d.geometry import BoundingBoxSE3, PoseSE3
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
)
from py123d.parser.griffin.griffin_map_parser import get_griffin_map_parsers, griffin_map_metadata
from py123d.parser.griffin.utils.griffin_constants import (
    GRIFFIN_BOX_DETECTION_FROM_STR,
    GRIFFIN_BOX_DETECTIONS_SE3_METADATA,
    GRIFFIN_CAMERA_HEIGHT,
    GRIFFIN_CAMERA_PERIOD_US,
    GRIFFIN_CAMERA_WIDTH,
    GRIFFIN_DRONE_AERIAL_MODALITY_ID,
    GRIFFIN_DRONE_AERIAL_STATIC_METADATA,
    GRIFFIN_DRONE_CAMERA_MAPPING,
    GRIFFIN_SPLITS,
    build_griffin_drone_ego_metadata,
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
    town_from_scene_name,
)

logger = logging.getLogger(__name__)

# Zero distortion: Griffin renders ideal pinhole cameras in CARLA.
_GRIFFIN_PINHOLE_DISTORTION = PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)


def _default_split_file(subset: str) -> Path:
    """Resolve the embedded official split file for ``subset``.

    The drone side reuses the *same* official train/val partitions as the
    vehicle side: Griffin's split files assign scenes by name, agent-agnostically.

    :param subset: Subset name, e.g. ``"griffin_50scenes_25m"``.
    :return: Path to the packaged ``splits/{subset}.json``.
    """
    return Path(str(resources.files("py123d.parser.griffin").joinpath("splits").joinpath(f"{subset}.json")))


class GriffinDroneParser(BaseDatasetParser):
    """Dataset parser for the Griffin dataset (drone-side, single agent)."""

    def __init__(
        self,
        splits: List[str],
        griffin_data_root: Union[Path, str],
        split_data_root: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initialize the :class:`GriffinDroneParser`.

        :param splits: Splits to convert, e.g. ``["griffin_50scenes_25m_train"]``.
            Each split's subset prefix selects the ``griffin-release`` tree and
            the official train/val partition.
        :param griffin_data_root: Directory containing the subset folders, laid
            out as ``{griffin_data_root}/{subset}/griffin-release/drone-side``
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

    def _drone_root(self, subset: str) -> Path:
        """Return the ``drone-side`` directory for ``subset``."""
        return self._griffin_data_root / subset / "griffin-release" / "drone-side"

    def get_log_parsers(self) -> List[GriffinDroneLogParser]:  # type: ignore[override]
        """Inherited, see superclass. One log per scene, routed to its split partition."""
        log_parsers: List[GriffinDroneLogParser] = []
        for split in self._splits:
            subset, kind = split_to_subset_and_kind(split)
            drone_root = self._drone_root(subset)
            if not drone_root.is_dir():
                logger.warning("Griffin subset '%s' not found at %s; skipping split '%s'.", subset, drone_root, split)
                continue

            scene_index = load_scene_index(drone_root)
            wanted_scenes = set(load_split_scene_names(self._split_file(subset), kind))
            scene_frames = {name: frames for name, frames in scene_index}

            missing = wanted_scenes - set(scene_frames)
            if missing:
                logger.warning(
                    "Split '%s' lists %d scene(s) absent from %s/scene_infos.json (e.g. %s).",
                    split,
                    len(missing),
                    drone_root,
                    sorted(missing)[0],
                )

            for scene_name, frames in scene_index:
                if scene_name not in wanted_scenes or not frames:
                    continue
                log_parsers.append(
                    GriffinDroneLogParser(
                        griffin_data_root=self._griffin_data_root,
                        subset=subset,
                        scene_name=scene_name,
                        frames=frames,
                        split=split,
                    )
                )
        return log_parsers

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass. One global map per Griffin CARLA town."""
        return get_griffin_map_parsers()


class GriffinDroneLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Griffin scene (drone-side)."""

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
        # all stored camera relative paths, so they resolve from a single
        # ``get_sensor_root("griffin") == griffin_data_root`` at read time.
        self._release_rel = Path(subset) / "griffin-release"
        self._drone_root = griffin_data_root / self._release_rel / "drone-side"
        self._calib_dir = self._drone_root / "calib"

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        _town = town_from_scene_name(self._scene_name)
        return LogMetadata(
            dataset="griffin",
            split=self._split,
            log_name=self._scene_name,
            # The CARLA town doubles as location, linking the log to its global
            # map at {maps_root}/griffin/griffin_{town}.arrow (nuScenes pattern).
            location=_town,
            # Attach the global map metadata so ``has_map`` / ``map_locations``
            # scene filters resolve for Griffin (map itself loads by location).
            map_metadata=griffin_map_metadata(_town),
        )

    def _build_camera_metadata(self) -> Dict[CameraID, PinholeCameraMetadata]:
        """Build pinhole camera metadata from the static per-camera calibration files."""
        camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
        for camera_name, camera_id in GRIFFIN_DRONE_CAMERA_MAPPING.items():
            intrinsic, extrinsic = load_calibration(self._calib_dir, camera_name)
            assert intrinsic is not None, f"Missing intrinsics for Griffin drone camera '{camera_name}'."
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
                # frame in 123D, so it maps directly to camera_to_imu_se3. For the
                # drone, the gimbal/mount orientation is baked into this extrinsic.
                camera_to_imu_se3=PoseSE3.from_transformation_matrix(extrinsic),
                is_undistorted=True,
            )
        return camera_metadata

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_metadata = build_griffin_drone_ego_metadata()
        camera_metadata = self._build_camera_metadata()
        aerial_metadata = CustomModalityMetadata(
            modality_id=GRIFFIN_DRONE_AERIAL_MODALITY_ID,
            metadata=dict(GRIFFIN_DRONE_AERIAL_STATIC_METADATA),
        )

        for frame in self._frames:
            timestamp = Timestamp.from_us(int(frame) * GRIFFIN_CAMERA_PERIOD_US)

            ego_pose = read_json(self._drone_root / "pose" / f"{frame}.json")
            ego_to_global = pose_dict_to_ego_to_global_se3(ego_pose)
            ego_state = EgoStateSE3.from_imu(
                imu_se3=ego_to_global,
                metadata=ego_metadata,
                timestamp=timestamp,
                dynamic_state_se3=None,
            )

            box_detections = self._extract_box_detections(frame, timestamp, ego_to_global)
            parsed_cameras = self._extract_cameras(frame, timestamp, ego_to_global, camera_metadata)
            aerial_state = self._extract_aerial_state(ego_pose, timestamp, aerial_metadata)

            yield ModalitiesSync(
                timestamp=timestamp,
                modalities=[ego_state, box_detections, *parsed_cameras, aerial_state],
            )

    def _extract_box_detections(self, frame: str, timestamp: Timestamp, ego_to_global: PoseSE3) -> BoxDetectionsSE3:
        """Extract 3D box detections for a frame, lifted from the drone ego to the global frame."""
        annotations = parse_label_file(self._drone_root / "label" / f"{frame}.txt")

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

    def _extract_cameras(
        self,
        frame: str,
        timestamp: Timestamp,
        ego_to_global: PoseSE3,
        camera_metadata: Dict[CameraID, PinholeCameraMetadata],
    ) -> List[ParsedCamera]:
        """Reference all available camera images for a frame, with global camera poses."""
        parsed_cameras: List[ParsedCamera] = []
        for camera_name, camera_id in GRIFFIN_DRONE_CAMERA_MAPPING.items():
            relative_path = self._release_rel / "drone-side" / "camera" / camera_name / f"{frame}.png"
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

    def _extract_aerial_state(
        self, ego_pose: Dict[str, float], timestamp: Timestamp, aerial_metadata: CustomModalityMetadata
    ) -> CustomModality:
        """Emit aerial-specific per-frame state for the UAV ego.

        Altitude is the ENU ``z`` of the drone ego pose (metres). The static
        "this ego is a UAV" facts live in ``aerial_metadata``; here we carry only
        what varies per frame.
        """
        aerial_data = {
            "altitude_m": float(ego_pose["z"]),
        }
        return CustomModality(data=aerial_data, metadata=aerial_metadata, timestamp=timestamp)
