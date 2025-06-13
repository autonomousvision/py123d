from pathlib import Path
from typing import Union

import pyarrow as pa


def open_arrow_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    with pa.memory_map(str(arrow_file_path), "rb") as source:
        table: pa.Table = pa.ipc.open_file(source).read_all()
    return table
