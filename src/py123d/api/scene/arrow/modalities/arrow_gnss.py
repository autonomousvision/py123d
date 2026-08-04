from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.gnss import Gnss, GnssMetadata
from py123d.datatypes.time.timestamp import Timestamp

_LLA_SIZE: int = 3
_COVARIANCE_SIZE: int = 9

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowGnssWriter(ArrowBaseModalityWriter):
    """Writes GNSS fixes to an Arrow IPC file.

    Geodetic coordinates are stored as one fixed-size (lat, lon, alt) column; the NavSatFix
    quality fields are nullable columns, so fixes lacking them are stored as null.
    """

    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, GnssMetadata), f"Expected GnssMetadata, got {type(metadata)}"

        self._metadata = metadata
        self._key = metadata.modality_key

        schema = pa.schema(
            [
                (f"{self._key}.timestamp_us", pa.int64()),
                (f"{self._key}.lla", pa.list_(pa.float64(), _LLA_SIZE)),
                (f"{self._key}.position_covariance", pa.list_(pa.float64(), _COVARIANCE_SIZE)),
                (f"{self._key}.position_covariance_type", pa.int32()),
                (f"{self._key}.status", pa.int32()),
                (f"{self._key}.service", pa.int32()),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=log_dir / f"{self._key}.arrow",
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, Gnss), f"Expected Gnss, got {type(modality)}"
        covariance = modality.position_covariance
        self.write_batch(
            {
                f"{self._key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._key}.lla": [modality.lla],
                f"{self._key}.position_covariance": [covariance if covariance is not None else None],
                f"{self._key}.position_covariance_type": [modality.position_covariance_type],
                f"{self._key}.status": [modality.status],
                f"{self._key}.service": [modality.service],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowGnssReader(ArrowBaseModalityReader):
    """Stateless reader for GNSS data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Gnss]:
        assert isinstance(metadata, GnssMetadata), f"Expected GnssMetadata, got {type(metadata)}"
        key = metadata.modality_key

        lla = table[f"{key}.lla"][index].as_py()
        covariance = table[f"{key}.position_covariance"][index].as_py()
        return Gnss(
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
            latitude=lla[0],
            longitude=lla[1],
            altitude=lla[2],
            position_covariance=np.asarray(covariance, dtype=np.float64) if covariance is not None else None,
            position_covariance_type=table[f"{key}.position_covariance_type"][index].as_py(),
            status=table[f"{key}.status"][index].as_py(),
            service=table[f"{key}.service"][index].as_py(),
        )

    @staticmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name not in table.column_names:
            raise ValueError(
                f"Column '{full_column_name}' not found in Arrow table for modality '{metadata.modality_key}'"
            )
        value = table[full_column_name][index].as_py()
        if deserialize and value is not None and column == "timestamp_us":
            value = Timestamp.from_us(value)
        return value
