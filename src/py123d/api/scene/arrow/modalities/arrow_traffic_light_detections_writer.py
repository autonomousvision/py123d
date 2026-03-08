from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetections, TrafficLightDetectionsMetadata


class ArrowTrafficLightDetectionsWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: TrafficLightDetectionsMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"

        schema = pa.schema(
            [
                (f"{self._modality_name}.timestamp_us", pa.int64()),
                (f"{self._modality_name}.lane_id", pa.list_(pa.int32())),
                (f"{self._modality_name}.status", pa.list_(pa.uint8())),
            ]
        )
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, traffic_light_detections: TrafficLightDetections):
        assert isinstance(traffic_light_detections, TrafficLightDetectionsMetadata), (
            f"Expected TrafficLightDetectionsMetadata, got {type(traffic_light_detections)}"
        )
        lane_id_list = []
        status_list = []

        for traffic_light_detection in traffic_light_detections:
            lane_id_list.append(traffic_light_detection.lane_id)
            status_list.append(traffic_light_detection.status)

        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [traffic_light_detections.timestamp.time_us],
                f"{self._modality_name}.lane_id": [lane_id_list],
                f"{self._modality_name}.status": [status_list],
            }
        )
