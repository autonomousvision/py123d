import unittest

from py123d.datatypes.time.time_point import TimePoint


class TestTimePoint(unittest.TestCase):

    def test_from_ns(self):
        """Test constructing TimePoint from nanoseconds."""
        tp = TimePoint.from_ns(1000000)
        assert tp.time_ns == 1000000
        assert tp.time_us == 1000

    def test_from_us(self):
        """Test constructing TimePoint from microseconds."""
        tp = TimePoint.from_us(1000)
        assert tp.time_us == 1000
        assert tp.time_ns == 1000000

    def test_from_ms(self):
        """Test constructing TimePoint from milliseconds."""
        tp = TimePoint.from_ms(1.5)
        assert tp.time_ms == 1.5
        assert tp.time_us == 1500

    def test_from_s(self):
        """Test constructing TimePoint from seconds."""
        tp = TimePoint.from_s(2.5)
        assert tp.time_s == 2.5
        assert tp.time_us == 2500000

    def test_time_ns_property(self):
        """Test accessing time value in nanoseconds."""
        tp = TimePoint.from_us(1000)
        assert tp.time_ns == 1000000

    def test_time_us_property(self):
        """Test accessing time value in microseconds."""
        tp = TimePoint.from_us(1000)
        assert tp.time_us == 1000

    def test_time_ms_property(self):
        """Test accessing time value in milliseconds."""
        tp = TimePoint.from_us(1500)
        assert tp.time_ms == 1.5

    def test_time_s_property(self):
        """Test accessing time value in seconds."""
        tp = TimePoint.from_us(2500000)
        assert tp.time_s == 2.5

    def test_from_ns_integer_assertion(self):
        """Test that from_ns raises AssertionError for non-integer input."""
        with self.assertRaises(AssertionError):
            TimePoint.from_ns(1000.5)

    def test_from_us_integer_assertion(self):
        """Test that from_us raises AssertionError for non-integer input."""
        with self.assertRaises(AssertionError):
            TimePoint.from_us(1000.5)

    def test_conversion_chain(self):
        """Test conversions between different time units."""
        original_us = 123456
        tp = TimePoint.from_us(original_us)
        assert TimePoint.from_ns(tp.time_ns).time_us == original_us
        assert TimePoint.from_ms(tp.time_ms).time_us == original_us
        assert TimePoint.from_s(tp.time_s).time_us == original_us
