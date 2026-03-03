from py123d.datatypes.detections import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus
from py123d.datatypes.time.timestamp import Timestamp

DUMMY_TIMESTAMP = Timestamp.from_s(0)


class TestTrafficLightStatus:
    def test_status_values(self):
        """Test that TrafficLightStatus enum has correct values."""
        assert TrafficLightStatus.GREEN.value == 0
        assert TrafficLightStatus.YELLOW.value == 1
        assert TrafficLightStatus.RED.value == 2
        assert TrafficLightStatus.OFF.value == 3
        assert TrafficLightStatus.UNKNOWN.value == 4


class TestTrafficLightDetection:
    def test_creation(self):
        """Test that TrafficLightDetection can be created with required fields."""
        detection = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)
        assert detection.lane_id == 1
        assert detection.status == TrafficLightStatus.GREEN


class TestTrafficLightDetections:
    def setup_method(self):
        self.det1 = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)
        self.det2 = TrafficLightDetection(lane_id=2, status=TrafficLightStatus.RED)
        self.det3 = TrafficLightDetection(lane_id=3, status=TrafficLightStatus.YELLOW)
        self.wrapper = TrafficLightDetections(detections=[self.det1, self.det2, self.det3], timestamp=DUMMY_TIMESTAMP)

    def test_getitem(self):
        """Test __getitem__ method of TrafficLightDetections."""
        assert self.wrapper[0] == self.det1
        assert self.wrapper[1] == self.det2
        assert self.wrapper[2] == self.det3

    def test_len(self):
        """Test __len__ method of TrafficLightDetections."""
        assert len(self.wrapper) == 3

    def test_iter(self):
        """Test __iter__ method of TrafficLightDetections."""
        detections = list(self.wrapper)
        assert detections == [self.det1, self.det2, self.det3]

    def test_get_by_lane_id_found(self):
        """Test get_by_lane_id method of TrafficLightDetections."""
        result = self.wrapper.get_by_lane_id(2)
        assert result == self.det2
        assert result.status == TrafficLightStatus.RED

    def test_get_by_lane_id_not_found(self):
        """Test get_by_lane_id method of TrafficLightDetections when not found."""
        result = self.wrapper.get_by_lane_id(99)
        assert result is None

    def test_get_by_lane_id_first_match(self):
        """Test get_by_lane_id method returns first match."""
        duplicate = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.OFF)
        wrapper = TrafficLightDetections(detections=[self.det1, duplicate], timestamp=DUMMY_TIMESTAMP)
        result = wrapper.get_by_lane_id(1)
        assert result == self.det1

    def test_empty_wrapper(self):
        """Test behavior of an empty TrafficLightDetections."""
        empty_wrapper = TrafficLightDetections(detections=[], timestamp=DUMMY_TIMESTAMP)
        assert len(empty_wrapper) == 0
        assert list(empty_wrapper) == []
        assert empty_wrapper.get_by_lane_id(1) is None

    def test_timestamp(self):
        """Test that TrafficLightDetections stores a timestamp."""
        timestamp = Timestamp.from_s(42)
        wrapper = TrafficLightDetections(detections=[self.det1], timestamp=timestamp)
        assert wrapper.timestamp == timestamp
