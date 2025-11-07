import unittest

from py123d.conversion.registry.box_detection_label_registry import BoxDetectionLabel, DefaultBoxDetectionLabel
from py123d.datatypes.detections import (
    BoxDetectionMetadata,
    BoxDetectionSE2,
    BoxDetectionSE3,
    BoxDetectionWrapper,
)
from py123d.datatypes.time.time_point import TimePoint
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, StateSE2, StateSE3, Vector2D, Vector3D


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


class TestBoxDetectionMetadata(unittest.TestCase):

    def test_initialization(self):
        metadata = BoxDetectionMetadata(**sample_metadata_args)
        self.assertIsInstance(metadata, BoxDetectionMetadata)
        self.assertEqual(metadata.label, DummyBoxDetectionLabel.CAR)
        self.assertEqual(metadata.track_token, "sample_token")
        self.assertEqual(metadata.num_lidar_points, 10)
        self.assertIsInstance(metadata.timepoint, TimePoint)

    def test_default_label(self):
        metadata = BoxDetectionMetadata(**sample_metadata_args)
        label = metadata.label
        default_label = metadata.default_label
        self.assertEqual(label, DummyBoxDetectionLabel.CAR)
        self.assertEqual(label.to_default(), DefaultBoxDetectionLabel.VEHICLE)
        self.assertEqual(default_label, DefaultBoxDetectionLabel.VEHICLE)

    def test_default_label_with_default_label(self):
        sample_args = sample_metadata_args.copy()
        sample_args["label"] = DefaultBoxDetectionLabel.PERSON
        metadata = BoxDetectionMetadata(**sample_args)
        label = metadata.label
        default_label = metadata.default_label
        self.assertEqual(label, DefaultBoxDetectionLabel.PERSON)
        self.assertEqual(default_label, DefaultBoxDetectionLabel.PERSON)

    def test_optional_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.BICYCLE,
            "track_token": "another_token",
        }
        metadata = BoxDetectionMetadata(**sample_args)
        self.assertIsInstance(metadata, BoxDetectionMetadata)
        self.assertEqual(metadata.label, DummyBoxDetectionLabel.BICYCLE)
        self.assertEqual(metadata.track_token, "another_token")
        self.assertIsNone(metadata.num_lidar_points)
        self.assertIsNone(metadata.timepoint)

    def test_missing_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.CAR,
        }
        with self.assertRaises(TypeError):
            BoxDetectionMetadata(**sample_args)

        sample_args = {
            "track_token": "token_only",
        }
        with self.assertRaises(TypeError):
            BoxDetectionMetadata(**sample_args)

        sample_args = {
            "timepoint": TimePoint.from_s(0.0),
        }
        with self.assertRaises(TypeError):
            BoxDetectionMetadata(**sample_args)


class TestBoxDetectionSE2(unittest.TestCase):

    def setUp(self):
        self.metadata = BoxDetectionMetadata(**sample_metadata_args)
        self.bounding_box_se2 = BoundingBoxSE2(
            center=StateSE2(x=0.0, y=0.0, yaw=0.0),
            length=4.0,
            width=2.0,
        )
        self.velocity = None

    def test_initialization(self):
        box_detection = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity=self.velocity,
        )
        self.assertIsInstance(box_detection, BoxDetectionSE2)
        self.assertEqual(box_detection.metadata, self.metadata)
        self.assertEqual(box_detection.bounding_box_se2, self.bounding_box_se2)
        self.assertIsNone(box_detection.velocity)

    def test_properties(self):
        box_detection = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity=self.velocity,
        )
        self.assertEqual(box_detection.shapely_polygon, self.bounding_box_se2.shapely_polygon)
        self.assertEqual(box_detection.center, self.bounding_box_se2.center)
        self.assertEqual(box_detection.bounding_box, self.bounding_box_se2)

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
        )
        self.assertIsInstance(box_detection_no_velo, BoxDetectionSE2)
        self.assertIsNone(box_detection_no_velo.velocity)

        box_detection_velo = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se2,
            velocity=Vector2D(x=1.0, y=0.0),
        )
        self.assertIsInstance(box_detection_velo, BoxDetectionSE2)
        self.assertEqual(box_detection_velo.velocity, Vector2D(x=1.0, y=0.0))


class TestBoxBoxDetectionSE3(unittest.TestCase):

    def setUp(self):
        self.metadata = BoxDetectionMetadata(**sample_metadata_args)
        self.bounding_box_se3 = BoundingBoxSE3(
            center=StateSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
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
        self.assertIsInstance(box_detection, BoxDetectionSE3)
        self.assertEqual(box_detection.metadata, self.metadata)
        self.assertEqual(box_detection.bounding_box_se3, self.bounding_box_se3)
        self.assertEqual(box_detection.velocity, self.velocity)

    def test_properties(self):
        box_detection = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=self.velocity,
        )
        self.assertEqual(box_detection.shapely_polygon, self.bounding_box_se3.shapely_polygon)
        self.assertEqual(box_detection.center, self.bounding_box_se3.center_se3)
        self.assertEqual(box_detection.center_se3, self.bounding_box_se3.center_se3)
        self.assertEqual(box_detection.bounding_box, self.bounding_box_se3)
        self.assertEqual(box_detection.bounding_box_se2, self.bounding_box_se3.bounding_box_se2)

    def test_box_detection_se2_conversion(self):
        box_detection = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        box_detection_se2 = box_detection.box_detection_se2
        self.assertIsInstance(box_detection_se2, BoxDetectionSE2)
        self.assertEqual(box_detection_se2.metadata, self.metadata)
        self.assertEqual(box_detection_se2.bounding_box_se2, self.bounding_box_se3.bounding_box_se2)
        self.assertEqual(box_detection_se2.velocity, Vector2D(x=1.0, y=0.0))

    def test_box_detection_se3_conversion(self):
        box_detection_se2 = BoxDetectionSE2(
            metadata=self.metadata,
            bounding_box_se2=self.bounding_box_se3.bounding_box_se2,
            velocity=Vector2D(x=1.0, y=0.0),
        )
        box_detection_se3 = BoxDetectionSE3(
            metadata=box_detection_se2.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector2D(x=1.0, y=0.0),
        )
        self.assertIsInstance(box_detection_se3, BoxDetectionSE3)
        self.assertEqual(box_detection_se3.metadata, box_detection_se2.metadata)
        self.assertEqual(box_detection_se3.bounding_box_se3, self.bounding_box_se3)
        self.assertEqual(box_detection_se3.velocity, Vector2D(x=1.0, y=0.0))

        box_detection_se3_converted = box_detection_se3.box_detection_se2
        self.assertIsInstance(box_detection_se3_converted, BoxDetectionSE2)
        self.assertEqual(box_detection_se3_converted.metadata, box_detection_se2.metadata)
        self.assertEqual(box_detection_se3_converted.bounding_box_se2, box_detection_se2.bounding_box_se2)
        self.assertEqual(box_detection_se3_converted.velocity, box_detection_se2.velocity)

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
        )
        self.assertIsInstance(box_detection_no_velo, BoxDetectionSE3)
        self.assertIsNone(box_detection_no_velo.velocity)

        box_detection_velo = BoxDetectionSE3(
            metadata=self.metadata,
            bounding_box_se3=self.bounding_box_se3,
            velocity=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        self.assertIsInstance(box_detection_velo, BoxDetectionSE3)
        self.assertEqual(box_detection_velo.velocity, Vector3D(x=1.0, y=0.0, z=0.0))


class TestBoxDetectionWrapper(unittest.TestCase):

    def setUp(self):
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
                center=StateSE2(x=0.0, y=0.0, yaw=0.0),
                length=4.0,
                width=2.0,
            ),
            velocity=Vector2D(x=1.0, y=0.0),
        )
        self.box_detection2 = BoxDetectionSE2(
            metadata=self.metadata2,
            bounding_box_se2=BoundingBoxSE2(
                center=StateSE2(x=5.0, y=5.0, yaw=0.0),
                length=1.0,
                width=0.5,
            ),
            velocity=Vector2D(x=0.5, y=0.5),
        )
        self.box_detection3 = BoxDetectionSE3(
            metadata=self.metadata3,
            bounding_box_se3=BoundingBoxSE3(
                center=StateSE3(x=10.0, y=10.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=2.0,
                width=1.0,
                height=1.5,
            ),
            velocity=Vector3D(x=0.0, y=1.0, z=0.0),
        )

    def test_initialization(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        self.assertIsInstance(wrapper, BoxDetectionWrapper)
        self.assertEqual(len(wrapper.box_detections), 2)

    def test_empty_initialization(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        self.assertIsInstance(wrapper, BoxDetectionWrapper)
        self.assertEqual(len(wrapper.box_detections), 0)

    def test_getitem(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        self.assertEqual(wrapper[0], self.box_detection1)
        self.assertEqual(wrapper[1], self.box_detection2)

    def test_getitem_out_of_range(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1])
        with self.assertRaises(IndexError):
            _ = wrapper[1]

    def test_len(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2, self.box_detection3])
        self.assertEqual(len(wrapper), 3)

    def test_len_empty(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        self.assertEqual(len(wrapper), 0)

    def test_iter(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        detections = list(wrapper)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0], self.box_detection1)
        self.assertEqual(detections[1], self.box_detection2)

    def test_get_detection_by_track_token_found(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2, self.box_detection3])
        detection = wrapper.get_detection_by_track_token("token2")
        self.assertIsNotNone(detection)
        self.assertEqual(detection, self.box_detection2)
        self.assertEqual(detection.metadata.track_token, "token2")

    def test_get_detection_by_track_token_not_found(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        detection = wrapper.get_detection_by_track_token("nonexistent_token")
        self.assertIsNone(detection)

    def test_get_detection_by_track_token_empty_wrapper(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        detection = wrapper.get_detection_by_track_token("token1")
        self.assertIsNone(detection)

    def test_occupancy_map(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        occupancy_map = wrapper.occupancy_map
        self.assertIsNotNone(occupancy_map)
        self.assertEqual(len(occupancy_map.geometries), 2)
        self.assertEqual(len(occupancy_map.ids), 2)
        self.assertIn("token1", occupancy_map.ids)
        self.assertIn("token2", occupancy_map.ids)

    def test_occupancy_map_cached(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection2])
        occupancy_map1 = wrapper.occupancy_map
        occupancy_map2 = wrapper.occupancy_map
        self.assertIs(occupancy_map1, occupancy_map2)

    def test_occupancy_map_empty(self):
        wrapper = BoxDetectionWrapper(box_detections=[])
        occupancy_map = wrapper.occupancy_map
        self.assertIsNotNone(occupancy_map)
        self.assertEqual(len(occupancy_map.geometries), 0)
        self.assertEqual(len(occupancy_map.ids), 0)

    def test_mixed_detection_types(self):
        wrapper = BoxDetectionWrapper(box_detections=[self.box_detection1, self.box_detection3])
        self.assertEqual(len(wrapper), 2)
        self.assertIsInstance(wrapper[0], BoxDetectionSE2)
        self.assertIsInstance(wrapper[1], BoxDetectionSE3)
