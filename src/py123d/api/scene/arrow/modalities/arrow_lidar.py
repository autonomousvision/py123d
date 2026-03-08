from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.lidar.draco_lidar_io import encode_point_cloud_3d_as_draco_binary
from py123d.common.io.lidar.ipc_lidar_io import (
    encode_point_cloud_3d_as_ipc_binary,
    encode_point_cloud_features_as_ipc_binary,
)
from py123d.common.io.lidar.laz_lidar_io import encode_point_cloud_3d_as_laz_binary
from py123d.common.io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.parser.abstract_dataset_parser import LidarData


class ArrowLidarWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: Union[LidarMetadata, LidarMergedMetadata],
        log_metadata: LogMetadata,
        lidar_store_option: Literal["path", "binary"],
        lidar_point_cloud_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]],
        lidar_point_feature_codec: Optional[Literal["ipc_zstd", "ipc_lz4", "ipc"]],  # None drops features.
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata)), (
            f"Expected LidarMetadata or LidarMergedMetadata, got {type(metadata)}"
        )
        assert lidar_store_option in {"path", "binary"}, f"Unsupported lidar store option: {lidar_store_option}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name
        self._log_metadata = log_metadata

        self._lidar_store_option = lidar_store_option
        self._lidar_point_cloud_codec = lidar_point_cloud_codec
        self._lidar_point_feature_codec = lidar_point_feature_codec

        file_path = log_dir / f"{metadata.modality_name}.arrow"

        schema_list = [
            (f"{metadata.modality_name}.start_timestamp_us", pa.int64()),
            (f"{metadata.modality_name}.end_timestamp_us", pa.int64()),
        ]
        if lidar_store_option == "binary":
            schema_list.append((f"{metadata.modality_name}.point_cloud_3d", pa.binary()))
            if lidar_point_feature_codec:
                schema_list.append((f"{metadata.modality_name}.point_cloud_features", pa.binary()))
        elif lidar_store_option == "path":
            schema_list.append((f"{metadata.modality_name}.data", pa.string()))
        else:
            raise ValueError(f"Unsupported lidar store option: {lidar_store_option}")

        schema = add_metadata_to_arrow_schema(pa.schema(schema_list), metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, lidar_data: LidarData) -> None:
        batch: Dict[str, Union[List[int], List[Optional[str]], List[Optional[bytes]]]] = {
            f"{self._modality_name}.start_timestamp_us": [lidar_data.start_timestamp.time_us],
            f"{self._modality_name}.end_timestamp_us": [lidar_data.end_timestamp.time_us],
        }

        if self._lidar_store_option == "path":
            data_path: Optional[str] = str(lidar_data.relative_path) if lidar_data.has_file_path else None
            batch[f"{self._modality_name}.data"] = [data_path]

        elif self._lidar_store_option == "binary":
            point_cloud_binary, features_binary = self._prepare_lidar_data(lidar_data)
            batch[f"{self._modality_name}.point_cloud_3d"] = [point_cloud_binary]
            if self._lidar_point_feature_codec:
                batch[f"{self._modality_name}.point_cloud_features"] = [features_binary]

        self.write_batch(batch)

    def _prepare_lidar_data(self, lidar_data: LidarData) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Load and/or encode the lidar data in binary for point cloud and features.

        :param lidar_data: Helper class referencing the lidar observation.
        :return: Tuple of (point_cloud_binary, point_cloud_features_binary)
        """
        # 1. Load point cloud and point features
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if lidar_data.has_point_cloud_3d:
            point_cloud_3d = lidar_data.point_cloud_3d
            point_cloud_features = lidar_data.point_cloud_features
        elif lidar_data.has_file_path:
            lidar_metadatas = (
                dict(self._modality_metadata) if isinstance(self._modality_metadata, LidarMergedMetadata) else None
            )
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                lidar_data.relative_path,  # type: ignore
                self._log_metadata,
                lidar_data.iteration,
                lidar_data.dataset_root,
                lidar_metadatas=lidar_metadatas,
            )
        else:
            raise ValueError("Lidar data must provide either point cloud data or a file path.")

        # 2. Compress point clouds with target codec
        point_cloud_3d_output: Optional[bytes] = None
        if point_cloud_3d is not None:
            codec = self._lidar_point_cloud_codec
            if codec == "draco":
                point_cloud_3d_output = encode_point_cloud_3d_as_draco_binary(point_cloud_3d)
            elif codec == "laz":
                point_cloud_3d_output = encode_point_cloud_3d_as_laz_binary(point_cloud_3d)
            elif codec == "ipc":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec=None)
            elif codec == "ipc_zstd":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="zstd")
            elif codec == "ipc_lz4":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="lz4")
            else:
                raise NotImplementedError(f"Unsupported lidar point cloud codec: {codec}")

        # 3. Compress point cloud features with target codec, if specified
        point_cloud_feature_output: Optional[bytes] = None
        if self._lidar_point_feature_codec is not None and point_cloud_features is not None:
            if self._lidar_point_feature_codec == "ipc":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(point_cloud_features, codec=None)
            elif self._lidar_point_feature_codec == "ipc_zstd":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="zstd"
                )
            elif self._lidar_point_feature_codec == "ipc_lz4":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="lz4"
                )
            else:
                raise NotImplementedError(f"Unsupported lidar point feature codec: {self._lidar_point_feature_codec}")

        return point_cloud_3d_output, point_cloud_feature_output
