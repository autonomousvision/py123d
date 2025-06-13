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
        table_data = memoryview(self._mmap)[offset : offset + size]
        reader = ipc.open_file(pa.py_buffer(table_data))
        return reader.read_all()

    def list_tables(self):
        return list(self._tables.keys())

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_mmap"):
            self._mmap.close()
        if hasattr(self, "_file"):
            self._file.close()


def save_arrow_tables(tables_dict, filename):
    with open(filename, "wb") as f:
        # Write header: number of tables
        f.write(struct.pack("<I", len(tables_dict)))

        # Write table of contents (TOC)
        toc_start = f.tell()
        toc_size = len(tables_dict) * (4 + 8 + 8)  # name_len + offset + size
        f.seek(toc_start + toc_size)  # Skip TOC space for now

        toc_entries = []
        for name, table in tables_dict.items():
            offset = f.tell()

            # Write table using Arrow IPC
            sink = pa.BufferOutputStream()
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

            buffer = sink.getvalue()
            f.write(buffer.to_pybytes())

            toc_entries.append((name, offset, len(buffer)))

        # Write TOC at the beginning
        current_pos = f.tell()
        f.seek(toc_start)
        for name, offset, size in toc_entries:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", size))

        f.seek(current_pos)


def load_arrow_table(filename, table_name):
    with open(filename, "rb") as f:
        # Read number of tables
        num_tables = struct.unpack("<I", f.read(4))[0]

        # Read TOC to find our table
        for _ in range(num_tables):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            offset = struct.unpack("<Q", f.read(8))[0]
            size = struct.unpack("<Q", f.read(8))[0]

            if name == table_name:
                # Memory map the specific table section
                with open(filename, "rb") as mf:
                    with mmap.mmap(mf.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        table_data = mm[offset : offset + size]
                        reader = ipc.open_file(pa.py_buffer(table_data))
                        return reader.read_all()

    raise KeyError(f"Table '{table_name}' not found")


def list_tables(filename):
    with open(filename, "rb") as f:
        num_tables = struct.unpack("<I", f.read(4))[0]
        tables = []

        for _ in range(num_tables):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            f.read(16)  # Skip offset and size
            tables.append(name)

        return tables
