import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import pyarrow as pa

from d123.common.utils.arrow_helper import get_lru_cached_arrow_table
from d123.datatypes.scene.scene_metadata import LogMetadata


@lru_cache(maxsize=10000)
def get_log_metadata_from_arrow(arrow_file_path: Union[Path, str]) -> LogMetadata:
    table = get_lru_cached_arrow_table(arrow_file_path)
    log_metadata = LogMetadata.from_dict(json.loads(table.schema.metadata[b"log_metadata"].decode()))
    return log_metadata


def add_log_metadata_to_arrow_schema(schema: pa.schema, log_metadata: LogMetadata) -> pa.schema:
    schema = schema.with_metadata({"log_metadata": json.dumps(log_metadata.to_dict())})
    return schema
