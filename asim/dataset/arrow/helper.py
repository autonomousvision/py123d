from pathlib import Path
from typing import Union

import pyarrow as pa


def open_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    with pa.memory_map(str(arrow_file_path), "rb") as source:
        table: pa.Table = pa.ipc.open_file(source).read_all()
    return table


# def write_arrow_table(table: pa.Table, arrow_file_path: Union[str, Path]) -> None:
#     with pa.ipc.new_file(str(arrow_file_path), table.schema) as writer:
#         writer.write_table(table)


def write_arrow_table(table: pa.Table, arrow_file_path: Union[str, Path]) -> None:
    with pa.OSFile(str(arrow_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
