"""Roundtrip tests for BoxDetectionsSE3 writer and reader."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import (
    ArrowBoxDetectionsSE3Reader,
    ArrowBoxDetectionsSE3Writer,
)
from py123d.datatypes import Timestamp
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3, BoxDetectionsSE3
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D

from ..conftest import make_box_detections_metadata


def _make_detections(ts_us: int, num_boxes: int = 2) -> BoxDetectionsSE3:
    metadata = make_box_detections_metadata()
    boxes: List[BoxDetectionSE3] = []
    for j in range(num_boxes):
        boxes.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=DefaultBoxDetectionLabel.VEHICLE,
                    track_token=f"track_{j}",
                    num_lidar_points=100 + j,
                ),
                bounding_box_se3=BoundingBoxSE3(
                    center_se3=PoseSE3(x=float(j), y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                    length=4.5,
                    width=2.0,
                    height=1.6,
                ),
                velocity_3d=Vector3D(1.0, 0.0, 0.0),
            )
        )
    return BoxDetectionsSE3(box_detections=boxes, timestamp=Timestamp.from_us(ts_us), metadata=metadata)


class TestBoxDetectionsRoundtrip:
    def _write_and_read(self, log_dir, detections_list):
        metadata = make_box_detections_metadata()
        writer = ArrowBoxDetectionsSE3Writer(log_dir=log_dir, metadata=metadata)
        for det in detections_list:
            writer.write_modality(det)
        writer.close()

        import pyarrow as pa

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame_with_boxes(self, tmp_path: Path):
        det = _make_detections(1000, num_boxes=3)
        table = self._write_and_read(tmp_path, [det])
        assert table.num_rows == 1

        metadata = make_box_detections_metadata()
        result = ArrowBoxDetectionsSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert len(result.box_detections) == 3
        assert result.timestamp.time_us == 1000
        assert result.box_detections[0].attributes.track_token == "track_0"
        assert result.box_detections[0].attributes.label == DefaultBoxDetectionLabel.VEHICLE

    def test_empty_detections(self, tmp_path: Path):
        det = _make_detections(1000, num_boxes=0)
        table = self._write_and_read(tmp_path, [det])

        metadata = make_box_detections_metadata()
        result = ArrowBoxDetectionsSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert len(result.box_detections) == 0

    def test_velocity_none(self, tmp_path: Path):
        metadata = make_box_detections_metadata()
        box = BoxDetectionSE3(
            attributes=BoxDetectionAttributes(
                label=DefaultBoxDetectionLabel.VEHICLE, track_token="t1", num_lidar_points=10
            ),
            bounding_box_se3=BoundingBoxSE3(center_se3=PoseSE3.identity(), length=4.0, width=2.0, height=1.5),
            velocity_3d=None,
        )
        det = BoxDetectionsSE3(box_detections=[box], timestamp=Timestamp.from_us(0), metadata=metadata)
        table = self._write_and_read(tmp_path, [det])

        result = ArrowBoxDetectionsSE3Reader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.box_detections[0].velocity_3d is None

    def test_read_column_deserialize_unknown_warns(self, tmp_path: Path):
        det = _make_detections(1000)
        table = self._write_and_read(tmp_path, [det])
        metadata = make_box_detections_metadata()

        with pytest.warns(UserWarning, match="Deserialization for column 'track_token'"):
            ArrowBoxDetectionsSE3Reader.read_column_at_index(
                0, table, metadata, "track_token", "test-dataset", deserialize=True
            )
