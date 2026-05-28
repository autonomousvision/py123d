from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset

from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.datatypes import CameraID, LidarID
from py123d.geometry import PoseSE3Index
from py123d.geometry.transform import abs_to_rel_se3_array

Sample = Dict[str, torch.Tensor]


class Torch123Dataset(Dataset[Sample]):
    """A ``torch.utils.data.Dataset`` view over py123d scenes.

    Every sample is one scene: the full ego trajectory across all iterations, plus the
    front camera image and a BEV lidar raster taken at iteration ``0``. Missing sensors
    are filled with zeros and flagged via ``*_valid`` masks, so batches stack with the
    default torch collate.
    """

    def __init__(
        self,
        scenes: List[SceneAPI],
        camera_id: CameraID = CameraID.PCAM_F0,
        lidar_id: LidarID = LidarID.LIDAR_MERGED,
        image_hw: Tuple[int, int] = (224, 224),
        lidar_raster_hw: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize the dataset.

        :param scenes: List of scenes to wrap. Typically retrieved via ``get_filtered_scenes()``.
        :param camera_id: Camera channel to read at iteration 0.
        :param lidar_id: Lidar channel to read at iteration 0.
        :param image_hw: Target ``(height, width)`` for the camera image. Images are resized
            so different datasets and sensors yield uniformly-shaped batches; missing cameras
            return a zero tensor of the same shape.
        :param lidar_raster_hw: Target ``(height, width)`` for the BEV lidar raster. At the
            default ``pixel_size=0.25`` (4 px/m, matching Transfuser) this covers
            ``height * 0.25`` by ``width * 0.25`` meters around the lidar sensor.
        """

        self._scenes = scenes
        self._camera_id = camera_id
        self._lidar_id = lidar_id
        self._image_hw = image_hw
        self._lidar_raster_hw = lidar_raster_hw

    @property
    def num_scenes(self) -> int:
        """Number of underlying scenes (one sample per scene)."""
        return len(self._scenes)

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> Sample:
        scene = self._scenes[idx]

        # Ego trajectory across all iterations of the scene.
        trajectory, trajectory_valid = _get_ego_trajectory(scene)

        # Front camera at iteration 0, resized to a fixed shape so batches stack.
        camera_image, camera_valid = _get_camera(scene, self._camera_id, target_hw=self._image_hw)

        # Lidar at iteration 0, rasterized into a fixed BEV histogram so batches stack.
        lidar_raster, lidar_valid = _get_lidar_raster(scene, self._lidar_id, target_hw=self._lidar_raster_hw)

        sample: Sample = {
            "trajectory": trajectory,
            "trajectory_valid": trajectory_valid,
            "camera_image": camera_image,
            "camera_valid": camera_valid,
            "lidar_raster": lidar_raster,
            "lidar_valid": lidar_valid,
            "scene_idx": torch.tensor(idx, dtype=torch.long),
        }
        return sample


def _get_ego_trajectory(scene: SceneAPI) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to extract the ego trajectory from a scene, expressed in the ego frame of iteration 0.

    Each iteration's world-frame ``imu_se3`` pose is read and then transformed into the local
    frame of the iteration-0 IMU pose, so trajectories from different logs become directly
    comparable (the trajectory always starts at the origin). Iterations where the ego state is
    missing keep a zero pose and are flagged in the ``trajectory_valid`` mask.

    :param scene: The scene to read from. Iteration 0 must have a valid ego state.
    :return: ``(trajectory, trajectory_valid)`` where ``trajectory`` is a
        ``(T, 7)`` float32 tensor (xyz + scalar-first quaternion ``[qw, qx, qy, qz]``) and
        ``trajectory_valid`` is a ``(T,)`` bool tensor.
    """

    num_iterations = scene.number_of_iterations
    trajectory = np.zeros((num_iterations, len(PoseSE3Index)), dtype=np.float64)
    trajectory_valid = torch.zeros((num_iterations,), dtype=torch.bool)

    initial_ego_state = scene.get_ego_state_se3_at_iteration(iteration=0)
    assert initial_ego_state is not None, "Ego state at iteration 0 is required to determine the trajectory shape."
    for iteration in range(num_iterations):
        ego_state = scene.get_ego_state_se3_at_iteration(iteration=iteration)
        if ego_state is not None:
            trajectory[iteration] = ego_state.imu_se3.array
        trajectory_valid[iteration] = ego_state is not None

    trajectory = abs_to_rel_se3_array(origin=initial_ego_state.imu_se3, pose_se3_array=trajectory)
    trajectory = torch.from_numpy(trajectory.astype(np.float32, copy=False))
    return trajectory, trajectory_valid


def _get_camera(
    scene: SceneAPI,
    camera_id: CameraID,
    target_hw: Tuple[int, int],
    scale: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to extract and resize the camera image at iteration 0.

    The native camera image is resized to ``target_hw`` with area interpolation so batches
    stack uniformly across datasets and sensor resolutions. If the camera is not available
    at iteration 0, a zero tensor of the same shape is returned and ``valid`` is set to False.

    :param scene: The scene to read from.
    :param camera_id: Camera channel to read at iteration 0.
    :param target_hw: Target ``(height, width)`` of the output image.
    :param scale: Optional integer downscale denominator forwarded to the SceneAPI
        (e.g. ``2`` for half-size reads). Independent of the post-read resize.
    :return: ``(image, valid)``: a ``(H, W, 3)`` uint8 tensor and a scalar bool mask.
    """

    camera = scene.get_camera_at_iteration(iteration=0, camera_id=camera_id, scale=scale)
    if camera is None:
        image = torch.zeros((*target_hw, 3), dtype=torch.uint8)
        valid = torch.tensor(False)
    else:
        target_height, target_width = target_hw
        resized = cv2.resize(camera.image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.ascontiguousarray(resized))
        valid = torch.tensor(True)

    return image, valid


def _get_lidar_raster(
    scene: SceneAPI,
    lidar_id: LidarID,
    target_hw: Tuple[int, int],
    pixel_size: float = 0.25,
    lidar_split_height: float = 0.2,
    max_height_lidar: float = 100.0,
    hist_max_per_pixel: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper that rasterizes a lidar point cloud into a BEV histogram, transfuser-style.

    Points are filtered to ``lidar_split_height < z < max_height_lidar`` (above ground but
    below an overhead cutoff), splatted into an XY 2D histogram centered on the lidar sensor,
    clipped per pixel, and normalized to ``[0, 1]``. The output has shape ``(1, H, W)`` so it
    stacks with the default torch collate.

    :param scene: The scene to read from (iteration ``0``).
    :param lidar_id: Lidar channel to read.
    :param target_hw: ``(height, width)`` of the BEV raster in pixels.
    :param pixel_size: Meters per pixel. Default ``0.25`` matches Transfuser (4 px/m).
    :param lidar_split_height: Lower z-cutoff to drop ground returns.
    :param max_height_lidar: Upper z-cutoff to drop overhead reflections.
    :param hist_max_per_pixel: Per-pixel point count clip used for normalization.
    :return: ``(raster, valid)``: a ``(1, H, W)`` float32 tensor and a scalar bool mask.
    """
    lidar = scene.get_lidar_at_iteration(iteration=0, lidar_id=lidar_id)
    if lidar is None:
        raster = torch.zeros((1, *target_hw), dtype=torch.float32)
        valid = torch.tensor(False)
    else:
        target_height, target_width = target_hw
        half_extent_x_m = (target_height * pixel_size) / 2.0
        half_extent_y_m = (target_width * pixel_size) / 2.0
        x_bins = np.linspace(-half_extent_x_m, half_extent_x_m, target_height + 1)
        y_bins = np.linspace(-half_extent_y_m, half_extent_y_m, target_width + 1)

        points = lidar.point_cloud_3d
        in_height_band = (points[:, 2] > lidar_split_height) & (points[:, 2] < max_height_lidar)
        points = points[in_height_band]

        hist, _ = np.histogramdd(points[:, :2], bins=(x_bins, y_bins))
        np.clip(hist, a_min=None, a_max=hist_max_per_pixel, out=hist)
        hist = hist.astype(np.float32, copy=False) / float(hist_max_per_pixel)

        raster = torch.from_numpy(np.ascontiguousarray(hist))[None]  # (1, H, W)
        valid = torch.tensor(True)

    return raster, valid


def visualize_batch(batch: Dict[str, torch.Tensor]) -> Figure:
    """Render a batch as a ``B x 3`` matplotlib grid: trajectory, camera, lidar.

    :param batch: Output of one ``DataLoader`` iteration over :class:`Torch123Dataset`.
    :return: The created matplotlib :class:`~matplotlib.figure.Figure`. The caller is
        responsible for calling ``plt.show()`` or ``fig.savefig(...)``.
    """
    batch_size = batch["trajectory"].shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 3 * batch_size), squeeze=False)

    for b in range(batch_size):
        trajectory = batch["trajectory"][b].numpy()
        trajectory_valid = batch["trajectory_valid"][b].numpy().astype(bool)
        camera_image = batch["camera_image"][b].numpy()
        camera_valid = bool(batch["camera_valid"][b].item())
        lidar_raster = batch["lidar_raster"][b, 0].numpy()
        lidar_valid = bool(batch["lidar_valid"][b].item())
        scene_idx = int(batch["scene_idx"][b].item())

        # Column 0: ego trajectory in the ego frame of iteration 0 (starts at the origin).
        ax_traj = axes[b, 0]
        valid_xy = trajectory[trajectory_valid][:, PoseSE3Index.XY]
        if len(valid_xy) > 0:
            ax_traj.plot(valid_xy[:, PoseSE3Index.X], valid_xy[:, PoseSE3Index.Y], "-o", markersize=3)
            ax_traj.scatter(
                valid_xy[0, PoseSE3Index.X], valid_xy[0, PoseSE3Index.Y], color="tab:green", label="start", zorder=5
            )
            ax_traj.scatter(
                valid_xy[-1, PoseSE3Index.X], valid_xy[-1, PoseSE3Index.Y], color="tab:red", label="end", zorder=5
            )
            ax_traj.legend(loc="best", fontsize=8)
        ax_traj.set_aspect("equal")
        ax_traj.set_xlim(-40, 40)
        ax_traj.set_ylim(-40, 40)
        ax_traj.set_title(f"Scene {scene_idx}: trajectory (x, y)")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.grid(True)

        # Column 1: front camera image.
        ax_cam = axes[b, 1]
        ax_cam.imshow(camera_image)
        ax_cam.set_title(f"Camera {'(valid)' if camera_valid else '(missing)'}")
        ax_cam.axis("off")

        # Column 2: BEV lidar raster. origin='lower' so x=forward points up the page.
        ax_lidar = axes[b, 2]
        ax_lidar.imshow(lidar_raster, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax_lidar.set_title(f"Lidar BEV {'(valid)' if lidar_valid else '(missing)'}")
        ax_lidar.axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    """Build the dataset, iterate two batches via a DataLoader, and visualize the first one."""

    scene_filter = SceneFilter(
        split_names=None,
        log_names=None,
        scene_uuids=None,
        target_iteration_duration_s=0.1,  # 10Hz sampling rate.
        future_duration_s=4.0,
        history_duration_s=0.5,
        required_scene_modalities=["ego_state_se3", "camera.pcam_f0", "lidar:any"],
    )
    scenes: List[SceneAPI] = get_filtered_scenes(scene_filter)
    dataset = Torch123Dataset(scenes)
    print(f"Found {len(dataset)} scenes.")

    # All output tensors have fixed shapes (missing sensors are zero-filled with a ``*_valid``
    # mask), so the default torch collate is sufficient - no custom ``collate_fn`` needed.
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    first_batch: Optional[Dict[str, torch.Tensor]] = None
    for batch_idx, batch in enumerate(loader):
        if first_batch is None:
            first_batch = batch
        print(f"\nBatch {batch_idx}:")
        for key, tensor in batch.items():
            print(f"  {key:18s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        for valid_key in ("trajectory_valid", "camera_valid", "lidar_valid"):
            valid = batch[valid_key]
            print(f"  {valid_key:18s} sum: {int(valid.sum())}/{valid.numel()}")
        if batch_idx + 1 >= 2:
            break

    if first_batch is not None:
        visualize_batch(first_batch)
        plt.show()


if __name__ == "__main__":
    main()
