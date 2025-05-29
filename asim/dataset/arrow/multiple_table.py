import mmap
import struct

import pyarrow as pa
import pyarrow.ipc as ipc


class ArrowMultiTableFile:
    def __init__(self, filename):
        self.filename = filename
        self._file = open(filename, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_toc()

    def _parse_toc(self):
        self._tables = {}
        self._mmap.seek(0)
        num_tables = struct.unpack("<I", self._mmap.read(4))[0]

        for _ in range(num_tables):
            name_len = struct.unpack("<I", self._mmap.read(4))[0]
            name = self._mmap.read(name_len).decode("utf-8")
            offset = struct.unpack("<Q", self._mmap.read(8))[0]
            size = struct.unpack("<Q", self._mmap.read(8))[0]
            self._tables[name] = (offset, size)

    def get_table(self, name):
        if name not in self._tables:
            raise KeyError(f"Table '{name}' not found")

        offset, size = self._tables[name]
        table_data = self._mmap[offset : offset + size]
        reader = ipc.open_file(pa.py_buffer(table_data))
        return reader.read_all()

    def list_tables(self):
        return list(self._tables.keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_mmap"):
            self._mmap.close()
        if hasattr(self, "_file"):
            self._file.close()
