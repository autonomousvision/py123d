from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


@dataclass
class LogMetadata:

    dataset: str
    location: str

    @classmethod
    def from_arrow_table(cls, table: pa.Table) -> LogMetadata:
        return cls(
            dataset=table["dataset"][0].as_py(),
            location=table["location"][0].as_py(),
        )
