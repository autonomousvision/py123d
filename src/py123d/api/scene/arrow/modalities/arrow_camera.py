from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)
from py123d.common.io.camera.png_camera_io import (
    encode_image_as_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.parser.abstract_dataset_parser import ParsedCamera


class ArrowPinholeCameraWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: PinholeCameraMetadata,
        data_codec: Literal["path", "jpeg_binary", "png_binary"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, PinholeCameraMetadata), f"Expected PinholeCameraMetadata, got {type(metadata)}"
        assert data_codec in {"path", "jpeg_binary", "png_binary"}, f"Unsupported data codec: {data_codec}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name
        self._data_codec = data_codec

        data_type = pa.binary() if data_codec in {"jpeg_binary", "png_binary"} else pa.string()
        max_batch_size = 10 if data_codec in {"jpeg_binary", "png_binary"} else 1000

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.data", data_type),
                (f"{metadata.modality_name}.state_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=max_batch_size,
        )

    def write_modality(self, camera_data: ParsedCamera):
        assert isinstance(camera_data, ParsedCamera), f"Expected CameraData, got {type(camera_data)}"
        if self._data_codec == "jpeg_binary":
            data = _get_jpeg_binary_from_camera_data(camera_data)
        elif self._data_codec == "png_binary":
            data = _get_png_binary_from_camera_data(camera_data)
        else:
            data = str(camera_data.relative_path)
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [camera_data.timestamp.time_us],
                f"{self._modality_name}.data": [data],
                f"{self._modality_name}.state_se3": [camera_data.extrinsic],
            }
        )


class ArrowFisheyeMEICameraWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: FisheyeMEICameraMetadata,
        data_codec: Literal["path", "jpeg_binary", "png_binary"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, FisheyeMEICameraMetadata), (
            f"Expected FisheyeMEICameraMetadata, got {type(metadata)}"
        )
        assert data_codec in {"path", "jpeg_binary", "png_binary"}, f"Unsupported data codec: {data_codec}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name
        self._data_codec = data_codec

        data_type = pa.binary() if data_codec in {"jpeg_binary", "png_binary"} else pa.string()
        max_batch_size = 10 if data_codec in {"jpeg_binary", "png_binary"} else 1000

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.data", data_type),
                (f"{metadata.modality_name}.state_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=max_batch_size,
        )

    def write_modality(self, camera_data: ParsedCamera):
        assert isinstance(camera_data, ParsedCamera), f"Expected CameraData, got {type(camera_data)}"
        if self._data_codec == "jpeg_binary":
            data = _get_jpeg_binary_from_camera_data(camera_data)
        elif self._data_codec == "png_binary":
            data = _get_png_binary_from_camera_data(camera_data)
        else:
            data = str(camera_data.relative_path)
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [camera_data.timestamp.time_us],
                f"{self._modality_name}.data": [data],
                f"{self._modality_name}.state_se3": [camera_data.extrinsic],
            }
        )


def _get_jpeg_binary_from_camera_data(camera_data: ParsedCamera) -> bytes:
    if camera_data.has_jpeg_binary:
        return camera_data.jpeg_binary  # type: ignore
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_jpeg_binary_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        numpy_image = load_image_from_png_file(absolute_path)
        return encode_image_as_jpeg_binary(numpy_image)
    elif camera_data.has_numpy_image:
        return encode_image_as_jpeg_binary(camera_data.numpy_image)  # type: ignore[arg-type]
    else:
        raise NotImplementedError("Camera data must provide jpeg_binary, numpy_image, or file path for binary storage.")


def _get_png_binary_from_camera_data(camera_data: ParsedCamera) -> bytes:
    if camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_png_binary_from_png_file(absolute_path)
    elif camera_data.has_numpy_image:
        return encode_image_as_png_binary(camera_data.numpy_image)  # type: ignore[arg-type]
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        numpy_image = load_image_from_jpeg_file(absolute_path)
        return encode_image_as_png_binary(numpy_image)
    elif camera_data.has_jpeg_binary:
        numpy_image = decode_image_from_jpeg_binary(camera_data.jpeg_binary)  # type: ignore[arg-type]
        return encode_image_as_png_binary(numpy_image)
    else:
        raise NotImplementedError("Camera data must provide png_binary, numpy_image, or file path for binary storage.")
