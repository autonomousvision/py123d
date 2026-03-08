from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.detections.box_detections import BoxDetectionsSE3
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.geometry.geometry_index import BoundingBoxSE3Index, Vector3DIndex


class ArrowBoxDetectionsSE3Writer(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BoxDetectionsSE3Metadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_name}.timestamp_us", pa.int64()),
                (f"{self._modality_name}.bounding_box_se3", pa.list_(pa.list_(pa.float64(), len(BoundingBoxSE3Index)))),
                (f"{self._modality_name}.track_token", pa.list_(pa.string())),
                (f"{self._modality_name}.label", pa.list_(pa.uint16())),
                (f"{self._modality_name}.velocity_3d", pa.list_(pa.list_(pa.float64(), len(Vector3DIndex)))),
                (f"{self._modality_name}.num_lidar_points", pa.list_(pa.int32())),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)

        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, box_detections_se3: BoxDetectionsSE3):
        assert isinstance(box_detections_se3, BoxDetectionsSE3), (
            f"Expected BoxDetectionsSE3, got {type(box_detections_se3)}"
        )
        bounding_box_se3_list = []
        token_list = []
        label_list = []
        velocity_3d_list = []
        num_lidar_points_list = []
        for box_detection in box_detections_se3:
            bounding_box_se3_list.append(box_detection.bounding_box_se3)
            token_list.append(box_detection.attributes.track_token)
            label_list.append(box_detection.attributes.label)
            velocity_3d_list.append(box_detection.velocity_3d)
            num_lidar_points_list.append(box_detection.attributes.num_lidar_points)

        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [box_detections_se3.timestamp.time_us],
                f"{self._modality_name}.bounding_box_se3": [bounding_box_se3_list],
                f"{self._modality_name}.track_token": [token_list],
                f"{self._modality_name}.label": [label_list],
                f"{self._modality_name}.velocity_3d": [velocity_3d_list],
                f"{self._modality_name}.num_lidar_points": [num_lidar_points_list],
            }
        )
