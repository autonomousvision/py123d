from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from py123d.common.dataset_paths import get_dataset_paths
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.lidar import LiDARType


def load_lidar_pcs_from_file(
    relative_path: Union[str, Path],
    log_metadata: LogMetadata,
    index: Optional[int] = None,
    sensor_root: Optional[Union[str, Path]] = None,
) -> Dict[LiDARType, npt.NDArray[np.float32]]:
    """Loads LiDAR point clouds from a file, based on the dataset specified in the log metadata.

    :param relative_path: Relative path to the LiDAR file.
    :param log_metadata: Metadata containing dataset information.
    :param index: Optional index for datasets that require it, defaults to None
    :param sensor_root: Optional root path for sensor data, defaults to None
    :raises NotImplementedError: If the dataset is not supported
    :return: Dictionary mapping LiDAR types to their point cloud numpy arrays
    """
    # NOTE @DanielDauner: This function is designed s.t. it can load multiple lidar types at the same time.
    # Several datasets (e.g., PandaSet, nuScenes) have multiple LiDAR sensors stored in one file.
    # Returning this as a dict allows us to handle this case without unnecessary io overhead.

    assert relative_path is not None, "Relative path to LiDAR file must be provided."
    if sensor_root is None:
        sensor_root = get_dataset_paths().get_sensor_root(log_metadata.dataset)
        assert sensor_root is not None, (
            f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}."
        )

    full_lidar_path = Path(sensor_root) / relative_path
    assert full_lidar_path.exists(), f"LiDAR file not found: {sensor_root} / {relative_path}"

    # NOTE: We move data specific import into if-else block, to avoid data specific import errors
    if log_metadata.dataset == "nuplan":
        from py123d.conversion.datasets.nuplan.nuplan_sensor_io import load_nuplan_lidar_pcs_from_file

        lidar_pcs_dict = load_nuplan_lidar_pcs_from_file(full_lidar_path)

    elif log_metadata.dataset == "av2-sensor":
        from py123d.conversion.datasets.av2.av2_sensor_io import load_av2_sensor_lidar_pcs_from_file

        lidar_pcs_dict = load_av2_sensor_lidar_pcs_from_file(full_lidar_path)

    elif log_metadata.dataset == "wod_perception":
        from py123d.conversion.datasets.wod.wod_perception_sensor_io import load_wod_perception_lidar_pcs_from_file

        assert index is not None, "Index must be provided for WOD Perception LiDAR loading."
        lidar_pcs_dict = load_wod_perception_lidar_pcs_from_file(full_lidar_path, index, keep_polar_features=False)

    elif log_metadata.dataset == "pandaset":
        from py123d.conversion.datasets.pandaset.pandaset_sensor_io import load_pandaset_lidars_pcs_from_file

        lidar_pcs_dict = load_pandaset_lidars_pcs_from_file(full_lidar_path, index)

    elif log_metadata.dataset == "kitti360":
        from py123d.conversion.datasets.kitti360.kitti360_sensor_io import load_kitti360_lidar_pcs_from_file

        lidar_pcs_dict = load_kitti360_lidar_pcs_from_file(full_lidar_path, log_metadata)

    elif log_metadata.dataset == "nuscenes":
        from py123d.conversion.datasets.nuscenes.nuscenes_sensor_io import load_nuscenes_lidar_pcs_from_file

        lidar_pcs_dict = load_nuscenes_lidar_pcs_from_file(full_lidar_path, log_metadata)

    else:
        raise NotImplementedError(f"Loading LiDAR data for dataset {log_metadata.dataset} is not implemented.")

    return lidar_pcs_dict
