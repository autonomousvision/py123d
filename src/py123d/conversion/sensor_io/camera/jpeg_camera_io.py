from pathlib import Path
from typing import Dict, Optional

from omegaconf import DictConfig
from pyparsing import Union

from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraMetadata
from py123d.script.utils.dataset_path_utils import get_dataset_paths

DATASET_PATHS: DictConfig = get_dataset_paths()
DATASET_SENSOR_ROOT: Dict[str, Path] = {
    "nuplan": DATASET_PATHS.nuplan_sensor_root,
    "av2-sensor": DATASET_PATHS.av2_sensor_data_root,
    "wopd": DATASET_PATHS.wopd_data_root,
    "pandaset": DATASET_PATHS.pandaset_data_root,
}


def load_image_from_jpeg_file(
    dataset_name: str,
    dataset_root: Path,
    relative_path: Union[str, Path],
    camera_metadata: PinholeCameraMetadata,
    iteration: Optional[int] = None,
) -> PinholeCamera:
    assert relative_path is not None, "Relative path to camera JPEG file must be provided."


def load_image_from_jpeg_binary(
    dataset_name: str,
    relative_path: Union[str, Path],
    pinhole_camera_metadata: PinholeCameraMetadata,
    iteration: Optional[int] = None,
) -> PinholeCamera:
    assert relative_path is not None, "Relative path to camera JPEG file must be provided."
    absolute_path = Path(dataset_name) / relative_path
    with open(absolute_path, "rb") as f:
        jpeg_binary = f.read()
    return PinholeCamera(metadata=pinhole_camera_metadata, jpeg_binary=jpeg_binary)
