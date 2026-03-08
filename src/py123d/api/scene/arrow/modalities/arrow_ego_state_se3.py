from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import PoseSE3Index


class ArrowEgoStateSE3Writer(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: EgoStateSE3Metadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, EgoStateSE3Metadata), f"Expected EgoStateSE3Metadata, got {type(metadata)}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
                (f"{metadata.modality_name}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
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

    def write_modality(self, ego_state_se3: EgoStateSE3):
        assert isinstance(ego_state_se3, EgoStateSE3), f"Expected EgoStateSE3, got {type(ego_state_se3)}"
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [ego_state_se3.timestamp.time_us],
                f"{self._modality_name}.imu_se3": [ego_state_se3.imu_se3],
                f"{self._modality_name}.dynamic_state_se3": [ego_state_se3.dynamic_state_se3],
            }
        )
