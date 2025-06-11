from __future__ import annotations

import json
from dataclasses import dataclass

import pyarrow as pa


@dataclass
class LogMetadata:

    dataset: str
    log_name: str
    location: str

    @classmethod
    def from_arrow_table(cls, table: pa.Table) -> LogMetadata:
        return cls(**json.loads(table.schema.metadata[b"log_metadata"].decode()))
