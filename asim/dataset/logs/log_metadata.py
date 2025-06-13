from __future__ import annotations

import json
from dataclasses import dataclass

import pyarrow as pa

import asim


@dataclass
class LogMetadata:

    dataset: str
    log_name: str
    location: str
    timestep_seconds: float

    map_has_z: bool
    version: str = str(asim.__version__)

    @classmethod
    def from_arrow_table(cls, table: pa.Table) -> LogMetadata:
        return cls(**json.loads(table.schema.metadata[b"log_metadata"].decode()))
