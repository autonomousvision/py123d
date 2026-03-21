"""Roundtrip tests for TrafficLightDetections writer and reader."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import (
    ArrowTrafficLightDetectionsReader,
    ArrowTrafficLightDetectionsWriter,
)
from py123d.datatypes import (
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightStatus,
)
from tests.unit.api.scene.conftest import make_traffic_light_metadata


def _make_tl(ts_us: int, num_lights: int = 2) -> TrafficLightDetections:
    metadata = make_traffic_light_metadata()
    detections = [TrafficLightDetection(lane_id=j + 1, status=TrafficLightStatus(j % 3)) for j in range(num_lights)]
    return TrafficLightDetections(detections=detections, timestamp=Timestamp.from_us(ts_us), metadata=metadata)


class TestTrafficLightRoundtrip:
    def _write_and_read(self, log_dir, tl_list):
        metadata = make_traffic_light_metadata()
        writer = ArrowTrafficLightDetectionsWriter(log_dir=log_dir, metadata=metadata)
        for tl in tl_list:
            writer.write_modality(tl)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame(self, tmp_path: Path):
        tl = _make_tl(1000, num_lights=3)
        table = self._write_and_read(tmp_path, [tl])
        assert table.num_rows == 1

        metadata = make_traffic_light_metadata()
        result = ArrowTrafficLightDetectionsReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.timestamp.time_us == 1000
        assert len(result.detections) == 3
        assert result.detections[0].lane_id == 1
        assert result.detections[0].status == TrafficLightStatus.GREEN

    def test_empty_detections(self, tmp_path: Path):
        tl = _make_tl(1000, num_lights=0)
        table = self._write_and_read(tmp_path, [tl])

        metadata = make_traffic_light_metadata()
        result = ArrowTrafficLightDetectionsReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert len(result.detections) == 0

    def test_read_column_status_deserialized(self, tmp_path: Path):
        tl = _make_tl(1000, num_lights=2)
        table = self._write_and_read(tmp_path, [tl])

        metadata = make_traffic_light_metadata()
        statuses = ArrowTrafficLightDetectionsReader.read_column_at_index(
            0, table, metadata, "status", "test-dataset", deserialize=True
        )
        assert isinstance(statuses, list)
        assert all(isinstance(s, TrafficLightStatus) for s in statuses)
