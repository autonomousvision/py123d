import pytest

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
from py123d.geometry.bounding_box import BoundingBoxSE2


class TestEgoStateSE2:
    def setup_method(self):
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

        assert ego_state.rear_axle_se2 == self.rear_axle_pose
        assert ego_state.vehicle_parameters == self.vehicle_params
        assert ego_state.dynamic_state_se2 == self.dynamic_state
        assert ego_state.timepoint == self.timepoint
        assert ego_state.tire_steering_angle == self.tire_steering_angle

    def test_from_rear_axle(self):
        """Test creating EgoStateSE2 from rear axle."""
        ego_state = EgoStateSE2.from_rear_axle(
            rear_axle_se2=self.rear_axle_pose,
            dynamic_state_se2=self.dynamic_state,
            vehicle_parameters=self.vehicle_params,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se2 == self.rear_axle_pose
        assert ego_state.vehicle_parameters == self.vehicle_params

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

        assert ego_state.rear_axle_se2 is not None
        assert ego_state.vehicle_parameters == self.vehicle_params

    def test_rear_axle_property(self):
        """Test rear_axle property."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)
        assert ego_state.rear_axle_se2 == self.rear_axle_pose

    def test_center_property(self):
        """Test center property calculation."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        center = ego_state.center_se2
        assert center is not None
        assert center.x == pytest.approx(self.vehicle_params.rear_axle_to_center_longitudinal)
        assert center.y == pytest.approx(0.0)
        assert center.yaw == pytest.approx(0.0)

    def test_bounding_box_property(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        bbox = ego_state.bounding_box_se2
        bbox_center = BoundingBoxSE2(ego_state.center_se2, self.vehicle_params.length, self.vehicle_params.width)
        assert bbox is not None
        assert bbox.length == self.vehicle_params.length
        assert bbox.width == self.vehicle_params.width
        assert ego_state.bounding_box_se2 == bbox_center

    def test_box_detection_property(self):
        """Test box detection properties."""
        ego_state = EgoStateSE2(
            rear_axle_se2=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se2=self.dynamic_state,
            timepoint=self.timepoint,
        )

        box_det = ego_state.box_detection_se2
        assert box_det is not None
        assert box_det.metadata.label == DefaultBoxDetectionLabel.EGO
        assert box_det.metadata.track_token == EGO_TRACK_TOKEN
        assert box_det.metadata.timepoint == self.timepoint

    def test_optional_parameters_none(self):
        """Test EgoStateSE2 with optional parameters as None."""
        ego_state = EgoStateSE2(
            rear_axle_se2=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se2=None,
            timepoint=None,
            tire_steering_angle=None,
        )

        assert ego_state.dynamic_state_se2 is None
        assert ego_state.timepoint is None
        assert ego_state.tire_steering_angle is None

    def test_default_tire_steering_angle(self):
        """Test default tire steering angle value."""
        ego_state = EgoStateSE2(rear_axle_se2=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        assert ego_state.tire_steering_angle == 0.0


class TestEgoStateSE3:
    def setup_method(self):
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

        assert ego_state.rear_axle_se3 == self.rear_axle_pose
        assert ego_state.vehicle_parameters == self.vehicle_params
        assert ego_state.dynamic_state_se3 == self.dynamic_state
        assert ego_state.timepoint == self.timepoint
        assert ego_state.tire_steering_angle == self.tire_steering_angle

    def test_from_rear_axle(self):
        """Test creating EgoStateSE3 from rear axle."""
        ego_state = EgoStateSE3.from_rear_axle(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
            tire_steering_angle=self.tire_steering_angle,
        )

        assert ego_state.rear_axle_se3 == self.rear_axle_pose
        assert ego_state.vehicle_parameters == self.vehicle_params

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

        assert ego_state.rear_axle_se3 is not None
        assert ego_state.vehicle_parameters == self.vehicle_params

    def test_rear_axle_properties(self):
        """Test rear_axle properties."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        assert ego_state.rear_axle_se3 == self.rear_axle_pose
        assert ego_state.rear_axle_se2 is not None

    def test_center_properties(self):
        """Test center property calculation."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        center = ego_state.center_se3
        assert center is not None
        assert center.x == pytest.approx(self.vehicle_params.rear_axle_to_center_longitudinal)
        assert center.y == pytest.approx(0.0)

        center_se2 = ego_state.center_se2
        assert center_se2 is not None

    def test_bounding_box_properties(self):
        """Test bounding box properties."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        bbox_se3 = ego_state.bounding_box_se3
        assert bbox_se3 is not None
        assert bbox_se3.length == self.vehicle_params.length
        assert bbox_se3.width == self.vehicle_params.width
        assert bbox_se3.height == self.vehicle_params.height

        bbox_se2 = ego_state.bounding_box_se2
        assert bbox_se2 is not None
        assert ego_state.bounding_box_se3 == bbox_se3

    def test_box_detection_properties(self):
        """Test box detection properties."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=self.dynamic_state,
            timepoint=self.timepoint,
        )

        box_det_se3 = ego_state.box_detection_se3
        assert box_det_se3 is not None
        assert box_det_se3.metadata.label == DefaultBoxDetectionLabel.EGO
        assert box_det_se3.metadata.track_token == EGO_TRACK_TOKEN
        assert box_det_se3.metadata.timepoint == self.timepoint

        box_det_se2 = ego_state.box_detection_se2
        assert box_det_se2 is not None
        assert box_det_se2.metadata.label == DefaultBoxDetectionLabel.EGO
        assert box_det_se2.metadata.track_token == EGO_TRACK_TOKEN
        assert box_det_se2.metadata.timepoint == self.timepoint

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
        assert ego_state_se2 is not None
        assert isinstance(ego_state_se2, EgoStateSE2)
        assert ego_state_se2.vehicle_parameters == self.vehicle_params
        assert ego_state_se2.timepoint == self.timepoint
        assert ego_state_se2.tire_steering_angle == self.tire_steering_angle

    def test_optional_parameters_none(self):
        """Test EgoStateSE3 with optional parameters as None."""
        ego_state = EgoStateSE3(
            rear_axle_se3=self.rear_axle_pose,
            vehicle_parameters=self.vehicle_params,
            dynamic_state_se3=None,
            timepoint=None,
            tire_steering_angle=None,
        )

        assert ego_state.dynamic_state_se3 is None
        assert ego_state.timepoint is None
        assert ego_state.tire_steering_angle is None

    def test_default_tire_steering_angle(self):
        """Test default tire steering angle value."""
        ego_state = EgoStateSE3(rear_axle_se3=self.rear_axle_pose, vehicle_parameters=self.vehicle_params)

        assert ego_state.tire_steering_angle == 0.0
