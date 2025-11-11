import json
from pathlib import Path
from typing import Union

import pyarrow as pa

from py123d.common.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.metadata.log_metadata import LogMetadata


def get_log_metadata_from_arrow_file(arrow_file_path: Union[Path, str]) -> LogMetadata:
    table = get_lru_cached_arrow_table(arrow_file_path)
    return get_log_metadata_from_arrow_table(table)


def get_log_metadata_from_arrow_table(arrow_table: pa.Table) -> LogMetadata:
    return LogMetadata.from_dict(json.loads(arrow_table.schema.metadata[b"log_metadata"].decode()))


def add_log_metadata_to_arrow_schema(schema: pa.schema, log_metadata: LogMetadata) -> pa.schema:
    schema = schema.with_metadata({"log_metadata": json.dumps(log_metadata.to_dict())})
    return schema
