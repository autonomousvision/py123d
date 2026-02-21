"""Integration tests for the modular folder-per-log Arrow storage.

Tests the full round-trip: ArrowLogWriter writes modality files -> ArrowSceneAPI reads them back.
"""

import tempfile
from pathlib import Path

import pytest

from py123d.api.scene.arrow.arrow_scene import ArrowSceneAPI
from py123d.common.utils.arrow_file_names import (
    BOX_DETECTIONS_FILE,
    EGO_STATE_FILE,
    INDEX_FILE,
    ROUTE_FILE,
    SCENARIO_TAGS_FILE,
    TRAFFIC_LIGHTS_FILE,
)
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.log_writer.arrow_log_writer import ArrowLogWriter
from py123d.conversion.registry.box_detection_label_registry import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import (
    TrafficLightDetection,
    TrafficLightDetectionWrapper,
    TrafficLightStatus,
)
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D


@pytest.fixture
def vehicle_parameters():
    return VehicleParameters(
        vehicle_name="test_vehicle",
        width=2.0,
        length=4.5,
        height=1.8,
        wheel_base=2.7,
        rear_axle_to_center_vertical=0.0,
        rear_axle_to_center_longitudinal=1.35,
    )


@pytest.fixture
def log_metadata(vehicle_parameters):
    return LogMetadata(
        dataset="test",
        split="test_train",
        log_name="test_log_001",
        location="test_city",
        timestep_seconds=0.1,
        vehicle_parameters=vehicle_parameters,
        box_detection_label_class=DefaultBoxDetectionLabel,
    )


@pytest.fixture
def config_ego_only():
    return DatasetConverterConfig(include_ego=True)


@pytest.fixture
def config_ego_and_detections():
    return DatasetConverterConfig(
        include_ego=True,
        include_box_detections=True,
        include_traffic_lights=True,
        include_scenario_tags=True,
        include_route=True,
    )


def _make_ego_state(timestamp_us: int, vehicle_params: VehicleParameters) -> EgoStateSE3:
    """Create a test EgoStateSE3."""
    x = float(timestamp_us) * 0.001
    rear_axle_se3 = PoseSE3(x=x, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    dynamic_state = DynamicStateSE3.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=rear_axle_se3,
        vehicle_parameters=vehicle_params,
        dynamic_state_se3=dynamic_state,
        timepoint=TimePoint.from_us(timestamp_us),
    )


def _make_box_detections(timestamp_us: int) -> BoxDetectionWrapper:
    """Create test box detections."""
    detections = [
        BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.VEHICLE,
                track_token="track_001",
                num_lidar_points=100,
                timepoint=TimePoint.from_us(timestamp_us),
            ),
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=10.0, y=5.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=4.0,
                width=2.0,
                height=1.5,
            ),
            velocity_3d=Vector3D.from_list([5.0, 0.0, 0.0]),
        ),
        BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                label=DefaultBoxDetectionLabel.PERSON,
                track_token="track_002",
                num_lidar_points=20,
                timepoint=TimePoint.from_us(timestamp_us),
            ),
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=20.0, y=-3.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=0.5,
                width=0.5,
                height=1.8,
            ),
            velocity_3d=Vector3D.from_list([0.0, 1.0, 0.0]),
        ),
    ]
    return BoxDetectionWrapper(box_detections=detections)


def _make_traffic_lights(timestamp_us: int) -> TrafficLightDetectionWrapper:
    """Create test traffic light detections."""
    return TrafficLightDetectionWrapper(
        traffic_light_detections=[
            TrafficLightDetection(
                timepoint=TimePoint.from_us(timestamp_us), lane_id=100, status=TrafficLightStatus.GREEN
            ),
            TrafficLightDetection(
                timepoint=TimePoint.from_us(timestamp_us), lane_id=101, status=TrafficLightStatus.RED
            ),
        ]
    )


class TestModularArrowWriteRead:
    """Test round-trip write/read for the modular Arrow storage."""

    def test_ego_state_only(self, log_metadata, config_ego_only, vehicle_parameters):
        """Write and read back ego state only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            # Write 3 keyframes
            assert writer.reset(config_ego_only, log_metadata) is True

            timestamps = [1000000, 1100000, 1200000]
            for ts in timestamps:
                ego = _make_ego_state(ts, vehicle_parameters)
                writer.write(timestamp=TimePoint.from_us(ts), ego_state=ego)
            writer.close()

            # Verify directory structure
            log_dir = logs_root / "test_train" / "test_log_001"
            assert log_dir.is_dir()
            assert (log_dir / INDEX_FILE).exists()
            assert (log_dir / EGO_STATE_FILE).exists()
            assert not (log_dir / BOX_DETECTIONS_FILE).exists()

            # Read back
            scene_api = ArrowSceneAPI(arrow_log_path=log_dir)

            # Check metadata
            assert scene_api.log_metadata.dataset == "test"
            assert scene_api.log_metadata.split == "test_train"
            assert scene_api.log_metadata.log_name == "test_log_001"
            assert scene_api.number_of_iterations == 3

            # Check ego state
            ego_0 = scene_api.get_ego_state_at_iteration(0)
            assert ego_0 is not None
            assert abs(ego_0.rear_axle_se3.x - 1000.0) < 1e-6

            ego_2 = scene_api.get_ego_state_at_iteration(2)
            assert ego_2 is not None
            assert abs(ego_2.rear_axle_se3.x - 1200.0) < 1e-6

            # Missing modalities return None
            assert scene_api.get_box_detections_at_iteration(0) is None
            assert scene_api.get_traffic_light_detections_at_iteration(0) is None
            assert scene_api.get_route_lane_group_ids(0) is None

    def test_ego_and_detections(self, log_metadata, config_ego_and_detections, vehicle_parameters):
        """Write and read back ego state, box detections, traffic lights, scenario tags, route."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            assert writer.reset(config_ego_and_detections, log_metadata) is True

            timestamps = [1000000, 1100000, 1200000]
            for ts in timestamps:
                ego = _make_ego_state(ts, vehicle_parameters)
                boxes = _make_box_detections(ts)
                tl = _make_traffic_lights(ts)
                writer.write(
                    timestamp=TimePoint.from_us(ts),
                    ego_state=ego,
                    box_detections=boxes,
                    traffic_lights=tl,
                    scenario_tags=["highway", "urban"],
                    route_lane_group_ids=[1, 2, 3],
                )
            writer.close()

            # Verify all files exist
            log_dir = logs_root / "test_train" / "test_log_001"
            assert (log_dir / INDEX_FILE).exists()
            assert (log_dir / EGO_STATE_FILE).exists()
            assert (log_dir / BOX_DETECTIONS_FILE).exists()
            assert (log_dir / TRAFFIC_LIGHTS_FILE).exists()
            assert (log_dir / SCENARIO_TAGS_FILE).exists()
            assert (log_dir / ROUTE_FILE).exists()

            # Read back
            scene_api = ArrowSceneAPI(arrow_log_path=log_dir)
            assert scene_api.number_of_iterations == 3

            # Box detections
            box_det = scene_api.get_box_detections_at_iteration(0)
            assert box_det is not None
            assert len(box_det) == 2

            # Traffic lights
            tl_det = scene_api.get_traffic_light_detections_at_iteration(1)
            assert tl_det is not None
            assert len(tl_det) == 2

            # Route
            route = scene_api.get_route_lane_group_ids(0)
            assert route == [1, 2, 3]

    def test_skip_existing_log(self, log_metadata, config_ego_only, vehicle_parameters):
        """Test that reset returns False when log already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            # First write
            assert writer.reset(config_ego_only, log_metadata) is True
            writer.write(timestamp=TimePoint.from_us(1000000), ego_state=_make_ego_state(1000000, vehicle_parameters))
            writer.close()

            # Second reset without force should return False
            assert writer.reset(config_ego_only, log_metadata) is False

    def test_force_reconversion(self, log_metadata, vehicle_parameters):
        """Test that force_log_conversion overwrites existing log."""
        config = DatasetConverterConfig(include_ego=True)
        config_force = DatasetConverterConfig(include_ego=True, force_log_conversion=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            # First write
            assert writer.reset(config, log_metadata) is True
            writer.write(timestamp=TimePoint.from_us(1000000), ego_state=_make_ego_state(1000000, vehicle_parameters))
            writer.close()

            # Second reset with force
            assert writer.reset(config_force, log_metadata) is True
            writer.write(timestamp=TimePoint.from_us(2000000), ego_state=_make_ego_state(2000000, vehicle_parameters))
            writer.close()

            # Should have overwritten data
            log_dir = logs_root / "test_train" / "test_log_001"
            scene_api = ArrowSceneAPI(arrow_log_path=log_dir)
            assert scene_api.number_of_iterations == 1

    def test_timepoint_access(self, log_metadata, config_ego_only, vehicle_parameters):
        """Test that timepoints are correctly stored and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            writer.reset(config_ego_only, log_metadata)
            timestamps = [1000000, 1100000, 1200000]
            for ts in timestamps:
                writer.write(timestamp=TimePoint.from_us(ts), ego_state=_make_ego_state(ts, vehicle_parameters))
            writer.close()

            log_dir = logs_root / "test_train" / "test_log_001"
            scene_api = ArrowSceneAPI(arrow_log_path=log_dir)

            for i, ts in enumerate(timestamps):
                tp = scene_api.get_timepoint_at_iteration(i)
                assert tp.time_us == ts

    def test_pickling(self, log_metadata, config_ego_only, vehicle_parameters):
        """Test that ArrowSceneAPI can be pickled and unpickled."""
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = Path(tmpdir) / "logs"
            writer = ArrowLogWriter(logs_root=logs_root)

            writer.reset(config_ego_only, log_metadata)
            writer.write(timestamp=TimePoint.from_us(1000000), ego_state=_make_ego_state(1000000, vehicle_parameters))
            writer.close()

            log_dir = logs_root / "test_train" / "test_log_001"
            scene_api = ArrowSceneAPI(arrow_log_path=log_dir)

            # Force scene metadata to be computed
            _ = scene_api.get_scene_metadata()

            # Pickle and unpickle
            pickled = pickle.dumps(scene_api)
            restored = pickle.loads(pickled)

            assert restored.log_metadata.dataset == "test"
            assert restored.number_of_iterations == 1
