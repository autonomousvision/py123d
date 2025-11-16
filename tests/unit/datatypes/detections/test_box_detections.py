import pytest

from py123d.conversion.registry.box_detection_label_registry import BoxDetectionLabel, DefaultBoxDetectionLabel
from py123d.datatypes.detections import (
    BoxDetectionMetadata,
    BoxDetectionSE2,
    BoxDetectionSE3,
    BoxDetectionWrapper,
)
from py123d.datatypes.time.time_point import TimePoint
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, PoseSE2, PoseSE3, Vector2D, Vector3D


class DummyBoxDetectionLabel(BoxDetectionLabel):

    CAR = 1
    PEDESTRIAN = 2
    BICYCLE = 3

    def to_default(self):
        mapping = {
            DummyBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            DummyBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            DummyBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]


sample_metadata_args = {
    "label": DummyBoxDetectionLabel.CAR,
    "track_token": "sample_token",
    "num_lidar_points": 10,
    "timepoint": TimePoint.from_s(0.0),
}


class TestBoxDetectionMetadata:

    def test_initialization(self):
        metadata = BoxDetectionMetadata(**sample_metadata_args)
        assert isinstance(metadata, BoxDetectionMetadata)
        assert metadata.label == DummyBoxDetectionLabel.CAR
        assert metadata.track_token == "sample_token"
        assert metadata.num_lidar_points == 10
        assert isinstance(metadata.timepoint, TimePoint)

    def test_default_label(self):
        metadata = BoxDetectionMetadata(**sample_metadata_args)
        label = metadata.label
        default_label = metadata.default_label
        assert label == DummyBoxDetectionLabel.CAR
        assert label.to_default() == DefaultBoxDetectionLabel.VEHICLE
        assert default_label == DefaultBoxDetectionLabel.VEHICLE

    def test_default_label_with_default_label(self):
        sample_args = sample_metadata_args.copy()
        sample_args["label"] = DefaultBoxDetectionLabel.PERSON
        metadata = BoxDetectionMetadata(**sample_args)
        label = metadata.label
        default_label = metadata.default_label
        assert label == DefaultBoxDetectionLabel.PERSON
        assert default_label == DefaultBoxDetectionLabel.PERSON

    def test_optional_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.BICYCLE,
            "track_token": "another_token",
        }
        metadata = BoxDetectionMetadata(**sample_args)
        assert isinstance(metadata, BoxDetectionMetadata)
        assert metadata.label == DummyBoxDetectionLabel.BICYCLE
        assert metadata.track_token == "another_token"
        assert metadata.num_lidar_points is None
        assert metadata.timepoint is None

    def test_missing_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.CAR,
        }
        with pytest.raises(TypeError):
            BoxDetectionMetadata(**sample_args)

        sample_args = {
            "track_token": "token_only",
        }
        with pytest.raises(TypeError):
            BoxDetectionMetadata(**sample_args)

        sample_args = {
            "timepoint": TimePoint.from_s(0.0),
        }
        with pytest.raises(TypeError):
            BoxDetectionMetadata(**sample_args)


class TestBoxDetectionSE2:

    def setup_method(self):
        self.metadata = BoxDetectionMetadata(**sample_metadata_args)
        self.bounding_box_se2 = BoundingBoxSE2(
            center_se2=PoseSE2(x=0.0, y=0.0, yaw=0.0),
            length=4.0,
            width=2.0,
        )
        self.velocity = None

    def test_initialization(self):
        box_detection = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.velocity,
        )
        assert isinstance(box_detection, BoxDetectionSE2)
        assert box_detection.metadata == self.metadata
        assert box_detection.bounding_box_se2 == self.bounding_box_se2
        assert box_detection.velocity_2d is None

    def test_properties(self):
        box_detection = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.velocity,
        )
        assert box_detection.shapely_polygon == self.bounding_box_se2.shapely_polygon
        assert box_detection.center_se2 == self.bounding_box_se2.center_se2
        assert box_detection.bounding_box_se2 == self.bounding_box_se2

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
        )
        assert isinstance(box_detection_no_velo, BoxDetectionSE2)
        assert box_detection_no_velo.velocity_2d is None

        box_detection_velo = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        assert isinstance(box_detection_velo, BoxDetectionSE2)
        assert box_detection_velo.velocity_2d == Vector2D(x=1.0, y=0.0)


class TestBoxBoxDetectionSE3:

    def setup_method(self):
        self.metadata = BoxDetectionMetadata(**sample_metadata_args)
        self.bounding_box_se3 = BoundingBoxSE3(
            center_se3=PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            length=4.0,
            width=2.0,
            height=1.5,
        )
        self.velocity = Vector3D(x=1.0, y=0.0, z=0.0)

    def test_initialization(self):
        box_detection = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=self.velocity,
        )
        assert isinstance(box_detection, BoxDetectionSE3)
        assert box_detection.metadata == self.metadata
        assert box_detection.bounding_box_se3 == self.bounding_box_se3
        assert box_detection.velocity_3d == self.velocity

    def test_properties(self):
        box_detection = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=self.velocity,
        )
        assert box_detection.shapely_polygon == self.bounding_box_se3.shapely_polygon
        assert box_detection.center_se3 == self.bounding_box_se3.center_se3
        assert box_detection.center_se2 == self.bounding_box_se3.center_se2
        assert box_detection.bounding_box_se3 == self.bounding_box_se3
        assert box_detection.bounding_box_se2 == self.bounding_box_se3.bounding_box_se2
        assert box_detection.velocity_3d == self.velocity
        assert box_detection.velocity_2d == self.velocity.vector_2d

    def test_box_detection_se2_conversion(self):
        box_detection = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        box_detection_se2 = box_detection.box_detection_se2
        assert isinstance(box_detection_se2, BoxDetectionSE2)
        assert box_detection_se2.metadata == self.metadata
        assert box_detection_se2.bounding_box_se2 == self.bounding_box_se3.bounding_box_se2
        assert box_detection_se2.velocity_2d == Vector2D(x=1.0, y=0.0)

    def test_box_detection_se3_conversion(self):
        box_detection_se2 = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se3.bounding_box_se2,
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        box_detection_se3 = BoxDetectionSE3(
            metadata=box_detection_se2.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        assert isinstance(box_detection_se3, BoxDetectionSE3)
        assert box_detection_se3.metadata == box_detection_se2.metadata
        assert box_detection_se3.bounding_box_se3 == self.bounding_box_se3
        assert box_detection_se3.velocity_2d == Vector2D(x=1.0, y=0.0)

        box_detection_se3_converted = box_detection_se3.box_detection_se2
        assert isinstance(box_detection_se3_converted, BoxDetectionSE2)
        assert box_detection_se3_converted.metadata == box_detection_se2.metadata
        assert box_detection_se3_converted.bounding_box_se2 == box_detection_se2.bounding_box_se2
        assert box_detection_se3_converted.velocity_2d == box_detection_se2.velocity_2d

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
        )
        assert isinstance(box_detection_no_velo, BoxDetectionSE3)
        assert box_detection_no_velo.velocity_3d is None

        box_detection_velo = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        assert isinstance(box_detection_velo, BoxDetectionSE3)
        assert box_detection_velo.velocity_3d == Vector3D(x=1.0, y=0.0, z=0.0)


class TestBoxDetectionWrapper:

    def setup_method(self):
        self.metadata1 = BoxDetectionMetadata(
            label=DummyBoxDetectionLabel.CAR,
            track_token="token1",
            num_lidar_points=10,
            timepoint=TimePoint.from_s(0.0),
        )
        self.metadata2 = BoxDetectionMetadata(
            label=DummyBoxDetectionLabel.PEDESTRIAN,
            track_token="token2",
            num_lidar_points=5,
            timepoint=TimePoint.from_s(0.0),
        )
        self.metadata3 = BoxDetectionMetadata(
            label=DummyBoxDetectionLabel.BICYCLE,
            track_token="token3",
            num_lidar_points=8,
            timepoint=TimePoint.from_s(0.0),
        )

        self.box_detection1 = BoxDetectionSE2(
            metadata=self.metadata1,
            bounding_box_se2=BoundingBoxSE2(
                center_se2=PoseSE2(x=0.0, y=0.0, yaw=0.0),
                length=4.0,
                width=2.0,
            ),
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        self.box_detection2 = BoxDetectionSE2(
            metadata=self.metadata2,
            bounding_box_se2=BoundingBoxSE2(
                center_se2=PoseSE2(x=5.0, y=5.0, yaw=0.0),
                length=1.0,
                width=0.5,
            ),
            velocity_2d=Vector2D(x=0.5, y=0.5),
        )
        self.box_detection3 = BoxDetectionSE3(
            metadata=self.metadata3,
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=10.0, y=10.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=2.0,
                width=1.0,
                height=1.5,
            ),
            velocity=Vector3D(x=0.0, y=1.0, z=0.0),
        )

    def test_initialization(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        assert isinstance(wrapper, BoxDetectionWrapper)
        assert len(wrapper.box_detections) == 2

    def test_empty_initialization(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        assert isinstance(wrapper, BoxDetectionWrapper)
        assert len(wrapper.box_detections) == 0

    def test_getitem(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        assert wrapper[0] == self.box_detection1
        assert wrapper[1] == self.box_detection2

    def test_getitem_out_of_range(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1])
        with pytest.raises(IndexError):
            _ = wrapper[1]

    def test_len(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2, self.box_detection3])
        assert len(wrapper) == 3

    def test_len_empty(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        assert len(wrapper) == 0

    def test_iter(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        detections = list(wrapper)
        assert len(detections) == 2
        assert detections[0] == self.box_detection1
        assert detections[1] == self.box_detection2

    def test_get_detection_by_track_token_found(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2, self.box_detection3])
        detection = wrapper.get_detection_by_track_token("token2")
        assert detection is not None
        assert detection == self.box_detection2
        assert detection.metadata.track_token == "token2"

    def test_get_detection_by_track_token_not_found(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        detection = wrapper.get_detection_by_track_token("nonexistent_token")
        assert detection is None

    def test_get_detection_by_track_token_empty_wrapper(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        detection = wrapper.get_detection_by_track_token("token1")
        assert detection is None

    def test_occupancy_map(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 2
        assert len(occupancy_map.ids) == 2
        assert "token1" in occupancy_map.ids
        assert "token2" in occupancy_map.ids

    def test_occupancy_map_cached(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        occupancy_map1 = wrapper.occupancy_map_2d
        occupancy_map2 = wrapper.occupancy_map_2d
        assert occupancy_map1 is occupancy_map2

    def test_occupancy_map_empty(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 0
        assert len(occupancy_map.ids) == 0

    def test_mixed_detection_types(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection3])
        assert len(wrapper) == 2
        assert isinstance(wrapper[0], BoxDetectionSE2)
        assert isinstance(wrapper[1], BoxDetectionSE3)
