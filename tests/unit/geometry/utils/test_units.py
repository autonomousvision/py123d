import pytest

from py123d.geometry.utils.units import kmph_to_mps, mph_to_mps, mps_to_kmph, mps_to_mph


class TestUnits:
    """Tests for unit conversion functions."""

    def test_mps_to_kmph(self):
        """Test converting meters per second to kilometers per hour."""
        assert mps_to_kmph(1.0) == pytest.approx(3.6)
        assert mps_to_kmph(0.0) == 0.0
        assert mps_to_kmph(10.0) == pytest.approx(36.0)

    def test_kmph_to_mps(self):
        """Test converting kilometers per hour to meters per second."""
        assert kmph_to_mps(3.6) == pytest.approx(1.0)
        assert kmph_to_mps(0.0) == 0.0
        assert kmph_to_mps(36.0) == pytest.approx(10.0)

    def test_mps_kmph_round_trip(self):
        """Test round-trip conversion between m/s and km/h."""
        for speed in [0.0, 1.0, 5.5, 27.78, 100.0]:
            assert kmph_to_mps(mps_to_kmph(speed)) == pytest.approx(speed)
            assert mps_to_kmph(kmph_to_mps(speed)) == pytest.approx(speed)

    def test_mph_to_mps(self):
        """Test converting miles per hour to meters per second."""
        assert mph_to_mps(1.0) == pytest.approx(0.44704)
        assert mph_to_mps(0.0) == 0.0
        assert mph_to_mps(60.0) == pytest.approx(26.8224)

    def test_mps_to_mph(self):
        """Test converting meters per second to miles per hour."""
        assert mps_to_mph(0.44704) == pytest.approx(1.0)
        assert mps_to_mph(0.0) == 0.0

    def test_mps_mph_round_trip(self):
        """Test round-trip conversion between m/s and mph."""
        for speed in [0.0, 1.0, 5.5, 27.78, 100.0]:
            assert mps_to_mph(mph_to_mps(speed)) == pytest.approx(speed)
            assert mph_to_mps(mps_to_mph(speed)) == pytest.approx(speed)
