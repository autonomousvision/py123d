import unittest

from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.time import TimePoint
from py123d.datatypes.vehicle_state import (
    DynamicStateSE2,
    DynamicStateSE3,
    EgoStateSE2,
    EgoStateSE3,
    VehicleParameters,
)
from py123d.datatypes.vehicle_state.ego_state import EGO_TRACK_TOKEN
from py123d.geometry import PoseSE2, PoseSE3, Vector2D, Vector3D


class TestEgoStateSE2(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for EgoStateSE2 tests."""
        self.rear_axle_pose = PoseSE2(x=0.0, y=0.0, yaw=0.0)
        self.vehicle_params = VehicleParameters(
            vehicle_name="test_vehicle",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            rear_axle_to_center_longitudinal=1.35,
            rear_axle_to_center_vertical=0.5,
        )
        self.dynamic_state = DynamicStateSE2(
            velocity=Vector2D(1.0, 0.0),
            acceleration=Vector2D(0.1, 0.0),
            angular_velocity=0.1,
        )
        self.timepoint = TimePoint.from_us(1000000)
        self.tire_steering_angle = 0.2

    def test_init(self):
        """Test EgoStateSE2 initialization."""
        ego_state = EgoStateSE2(
            rear_axle_se2=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se2=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertEqual(ego_state.rear_axle_se2, self.rear_axle_pose)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)
        self.assertEqual(ego_state.dynamic_state_se2, self.dynamic_state)
        self.assertEqual(ego_state.timepoint, self.timepoint)
        self.assertEqual(ego_state.tire_steering_angle, self.tire_steering_angle)

    def test_from_rear_axle(self):
        """Test creating EgoStateSE2 from rear axle."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            dynamic_state_se2=self.dynamic_state,
            vehicle_parameters=self.vehicle_params,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertEqual(ego_state.rear_axle_se2, self.rear_axle_pose)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)

    def test_from_center(self):
        """Test creating EgoStateSE2 from center pose."""
        center_pose = PoseSE2(x=1.35, y=0.0, yaw=0.0)
        ego_state = EgoStateSE2.from_center(
            center_se2=center_pose,
            dynamic_state_se2=self.dynamic_state,
            vehicle_parameters=self.vehicle_params,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertIsNotNone(ego_state.rear_axle_se2)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)

    def test_rear_axle_property(self):
        """Test rear_axle property."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        self.assertEqual(ego_state.rear_axle, self.rear_axle_pose)
        self.assertEqual(ego_state.rear_axle_se2, self.rear_axle_pose)

    def test_center_property(self):
        """Test center property calculation."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        center = ego_state.center_se2
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center.x, self.vehicle_params.rear_axle_to_center_longitudinal)
        self.assertAlmostEqual(center.y, 0.0)
        self.assertAlmostEqual(center.yaw, 0.0)

    def test_bounding_box_property(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        bbox = ego_state.bounding_box_se2
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox.length, self.vehicle_params.length)
        self.assertEqual(bbox.width, self.vehicle_params.width)
        self.assertEqual(ego_state.bounding_box, bbox)

    def test_box_detection_property(self):
        """Test box detection properties."""
        ego_state = EgoStateSE2(
            rear_axle_se2=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se2=self.dynamic_state,
            timepoint=self.timepoint,
        )

        box_det = ego_state.box_detection_se2
        self.assertIsNotNone(box_det)
        self.assertEqual(box_det.metadata.label, DefaultBoxDetectionLabel.EGO)
        self.assertEqual(box_det.metadata.track_token, EGO_TRACK_TOKEN)
        self.assertEqual(box_det.metadata.timepoint, self.timepoint)

        box_det = ego_state.box_detection
        self.assertIsNotNone(box_det)
        self.assertEqual(box_det.metadata.label, DefaultBoxDetectionLabel.EGO)
        self.assertEqual(box_det.metadata.track_token, EGO_TRACK_TOKEN)
        self.assertEqual(box_det.metadata.timepoint, self.timepoint)

    def test_optional_parameters_none(self):
        """Test EgoStateSE2 with optional parameters as None."""
        ego_state = EgoStateSE2(
            rear_axle_se2=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se2=None,
            timepoint=None,
            tire_steering_angle=None,
        )

        self.assertIsNone(ego_state.dynamic_state_se2)
        self.assertIsNone(ego_state.timepoint)
        self.assertIsNone(ego_state.tire_steering_angle)

    def test_default_tire_steering_angle(self):
        """Test default tire steering angle value."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        self.assertEqual(ego_state.tire_steering_angle, 0.0)


class TestEgoStateSE3(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for EgoStateSE3 tests."""

        self.rear_axle_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        self.vehicle_params = VehicleParameters(
            vehicle_name="test_vehicle",
            length=4.5,
            width=2.0,
            height=1.5,
            wheel_base=2.7,
            rear_axle_to_center_longitudinal=1.35,
            rear_axle_to_center_vertical=0.5,
        )
        self.dynamic_state = DynamicStateSE3(
            velocity=Vector3D(1.0, 0.0, 0.0),
            acceleration=Vector3D(0.1, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.1),
        )
        self.timepoint = TimePoint.from_us(1000000)
        self.tire_steering_angle = 0.2

    def test_init(self):
        """Test EgoStateSE3 initialization."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertEqual(ego_state.rear_axle_se3, self.rear_axle_pose)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)
        self.assertEqual(ego_state.dynamic_state_se3, self.dynamic_state)
        self.assertEqual(ego_state.timepoint, self.timepoint)
        self.assertEqual(ego_state.tire_steering_angle, self.tire_steering_angle)

    def test_from_rear_axle(self):
        """Test creating EgoStateSE3 from rear axle."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertEqual(ego_state.rear_axle_se3, self.rear_axle_pose)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)

    def test_from_center(self):
        """Test creating EgoStateSE3 from center pose."""
        center_pose = PoseSE3(x=1.35, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        ego_state = EgoStateSE3.from_center(
            center_se3=center_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        self.assertIsNotNone(ego_state.rear_axle_se3)
        self.assertEqual(ego_state.vehicle_parameters, self.vehicle_params)

    def test_rear_axle_properties(self):
        """Test rear_axle properties."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        self.assertEqual(ego_state.rear_axle, self.rear_axle_pose)
        self.assertEqual(ego_state.rear_axle_se3, self.rear_axle_pose)
        self.assertIsNotNone(ego_state.rear_axle_se2)

    def test_center_properties(self):
        """Test center property calculation."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        center = ego_state.center_se3
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center.x, self.vehicle_params.rear_axle_to_center_longitudinal)
        self.assertAlmostEqual(center.y, 0.0)

        center_se2 = ego_state.center_se2
        self.assertIsNotNone(center_se2)
        self.assertEqual(ego_state.center, ego_state.center_se3)

    def test_bounding_box_properties(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        bbox_se3 = ego_state.bounding_box_se3
        self.assertIsNotNone(bbox_se3)
        self.assertEqual(bbox_se3.length, self.vehicle_params.length)
        self.assertEqual(bbox_se3.width, self.vehicle_params.width)
        self.assertEqual(bbox_se3.height, self.vehicle_params.height)

        bbox_se2 = ego_state.bounding_box_se2
        self.assertIsNotNone(bbox_se2)
        self.assertEqual(ego_state.bounding_box, bbox_se3)

    def test_box_detection_properties(self):
        """Test box detection properties."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
        )

        box_det_se3 = ego_state.box_detection_se3
        self.assertIsNotNone(box_det_se3)
        self.assertEqual(box_det_se3.metadata.label, DefaultBoxDetectionLabel.EGO)
        self.assertEqual(box_det_se3.metadata.track_token, EGO_TRACK_TOKEN)
        self.assertEqual(box_det_se3.metadata.timepoint, self.timepoint)

        box_det_se2 = ego_state.box_detection_se2
        self.assertIsNotNone(box_det_se2)
        self.assertEqual(ego_state.box_detection, box_det_se3)

    def test_ego_state_se2_projection(self):
        """Test projection to EgoStateSE2."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        ego_state_se2 = ego_state.ego_state_se2
        self.assertIsNotNone(ego_state_se2)
        self.assertIsInstance(ego_state_se2, EgoStateSE2)
        self.assertEqual(ego_state_se2.vehicle_parameters, self.vehicle_params)
        self.assertEqual(ego_state_se2.timepoint, self.timepoint)
        self.assertEqual(ego_state_se2.tire_steering_angle, self.tire_steering_angle)

    def test_optional_parameters_none(self):
        """Test EgoStateSE3 with optional parameters as None."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=None,
            timepoint=None,
            tire_steering_angle=None,
        )

        self.assertIsNone(ego_state.dynamic_state_se3)
        self.assertIsNone(ego_state.timepoint)
        self.assertIsNone(ego_state.tire_steering_angle)

    def test_default_tire_steering_angle(self):
        """Test default tire steering angle value."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        self.assertEqual(ego_state.tire_steering_angle, 0.0)
