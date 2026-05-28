import numpy as np
import pytest
import shapely.geometry as geom

from py123d.geometry import Point2D, Point3D, Polyline2D, Polyline3D, PolylineSE2, PoseSE2


class TestPolyline2D:
    """Test class for Polyline2D."""

    def test_from_linestring(self):
        """Test creating Polyline2D from LineString."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        assert isinstance(polyline, Polyline2D)
        assert polyline.linestring.equals(linestring)

    def test_from_linestring_with_z(self):
        """Test creating Polyline2D from LineString with Z coordinates."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        assert isinstance(polyline, Polyline2D)
        assert not polyline.linestring.has_z

    def test_from_array_2d(self):
        """Test creating Polyline2D from 2D array."""
        array = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        polyline = Polyline2D.from_array(array)
        assert isinstance(polyline, Polyline2D)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_3d(self):
        """Test creating Polyline2D from 3D array."""
        array = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float32)
        polyline = Polyline2D.from_array(array)
        assert isinstance(polyline, Polyline2D)
        expected = array[:, :2]
        np.testing.assert_array_almost_equal(polyline.array, expected)

    def test_from_array_invalid_shape(self):
        """Test creating Polyline2D from invalid array shape."""
        array = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            Polyline2D.from_array(array)

    def test_array_property(self):
        """Test array property."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        array = polyline.array
        assert array.shape == (3, 2)
        assert array.dtype == np.float64
        np.testing.assert_array_almost_equal(array, coords)

    def test_length_property(self):
        """Test length property."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        assert polyline.length == 2.0

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = polyline.interpolate(1.0)
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 0.0

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        points = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 2)
        expected = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_array_almost_equal(points, expected)

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = polyline.interpolate(0.5, normalized=True)
        assert isinstance(point, Point2D)
        assert point.x == 1.0
        assert point.y == 0.0

    def test_project_point2d(self):
        """Test projecting Point2D onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        assert distance == 1.0

    def test_project_statese2(self):
        """Test projecting StateSE2 onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        state = PoseSE2(1.0, 1.0, 0.0)
        distance = polyline.project(state)
        assert distance == 1.0

    def test_project_shapely_point(self):
        """Test projecting a shapely Point onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = geom.Point(1.0, 0.5)
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_project_ndarray(self):
        """Test projecting a raw ndarray onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = np.array([1.0, 0.0])
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_from_array_invalid_last_dim(self):
        """Test from_array with invalid last dimension raises ValueError."""
        array = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            Polyline2D.from_array(array)

    def test_polyline_se2_property(self):
        """Test polyline_se2 property."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        polyline_se2 = polyline.polyline_se2
        assert isinstance(polyline_se2, PolylineSE2)

    def test_subline_returns_polyline2d(self):
        """subline returns a Polyline2D instance."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert isinstance(sub, Polyline2D)

    def test_subline_full_range(self):
        """subline(0, length) reproduces the original polyline."""
        array = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        polyline = Polyline2D.from_array(array)
        sub = polyline.subline(0.0, polyline.length)
        np.testing.assert_array_almost_equal(sub.array, array)
        assert sub.length == pytest.approx(polyline.length)

    def test_subline_partial_range(self):
        """subline cuts to a partial range with the expected length."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert sub.length == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(sub.array, [[1.0, 0.0], [3.0, 0.0]])

    def test_subline_exact_endpoints(self):
        """subline endpoints equal interpolate(start) / interpolate(end)."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]], dtype=np.float64))
        start_d, end_d = 1.0, polyline.length - 1.0
        sub = polyline.subline(start_d, end_d)
        np.testing.assert_array_almost_equal(sub.array[0], polyline.interpolate(start_d).array)
        np.testing.assert_array_almost_equal(sub.array[-1], polyline.interpolate(end_d).array)

    def test_subline_normalized(self):
        """Normalized distances produce the same result as absolute distances."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        sub_abs = polyline.subline(1.0, 3.0)
        sub_norm = polyline.subline(0.25, 0.75, normalized=True)
        np.testing.assert_array_almost_equal(sub_abs.array, sub_norm.array)

    def test_subline_clips_outside_range(self):
        """Distances outside [0, length] are clipped to the polyline bounds."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(-5.0, polyline.length + 5.0)
        np.testing.assert_array_almost_equal(sub.array, polyline.array)

    def test_subline_swaps_reversed(self):
        """Reversed (start > end) is silently swapped, matching shapely.ops.substring."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        np.testing.assert_array_almost_equal(
            polyline.subline(3.0, 1.0).array,
            polyline.subline(1.0, 3.0).array,
        )

    def test_subline_raises_on_zero_length(self):
        """subline raises ValueError when start_distance == end_distance after clipping."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        with pytest.raises(ValueError, match="start_distance != end_distance"):
            polyline.subline(2.0, 2.0)
        # Both ends clipped to the same boundary.
        with pytest.raises(ValueError):
            polyline.subline(-1.0, -2.0)

    def test_subline_preserves_intermediate_vertices(self):
        """Vertices strictly between start and end appear in the result."""
        array = np.array([[float(i), 0.0] for i in range(5)], dtype=np.float64)
        polyline = Polyline2D.from_array(array)
        sub = polyline.subline(0.5, 3.5)
        # Expect [(0.5, 0), (1, 0), (2, 0), (3, 0), (3.5, 0)]
        expected = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.5, 0.0]])
        np.testing.assert_array_almost_equal(sub.array, expected)

    def test_subline_does_not_call_shapely_substring(self, monkeypatch):
        """subline must not depend on shapely.ops.substring."""
        import shapely.ops

        def _fail(*args, **kwargs):
            raise RuntimeError("shapely.ops.substring should not be called by subline")

        monkeypatch.setattr(shapely.ops, "substring", _fail)
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert isinstance(sub, Polyline2D)


class TestPolylineSE2:
    """Test class for PolylineSE2."""

    def test_from_linestring(self):
        """Test creating PolylineSE2 from LineString."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = PolylineSE2.from_linestring(linestring)
        assert isinstance(polyline, PolylineSE2)
        assert polyline.array.shape == (3, 3)

    def test_from_array_2d(self):
        """Test creating PolylineSE2 from 2D array."""
        array = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        polyline = PolylineSE2.from_array(array)
        assert isinstance(polyline, PolylineSE2)
        assert polyline.array.shape == (3, 3)

    def test_from_array_se2(self):
        """Test creating PolylineSE2 from SE2 array."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        polyline = PolylineSE2.from_array(array)
        assert isinstance(polyline, PolylineSE2)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_invalid_shape(self):
        """Test creating PolylineSE2 from invalid array shape."""
        array = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            PolylineSE2.from_array(array)

    def test_length_property(self):
        """Test length property."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        assert polyline.length == 2.0

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = polyline.interpolate(1.0)
        assert isinstance(state, PoseSE2)
        assert state.x == 1.0
        assert state.y == 0.0

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        states = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        assert isinstance(states, np.ndarray)
        assert states.shape == (3, 3)

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = polyline.interpolate(0.5, normalized=True)
        assert isinstance(state, PoseSE2)
        assert state.x == 1.0
        assert state.y == 0.0

    def test_project_point2d(self):
        """Test projecting Point2D onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        assert distance == 1.0

    def test_project_statese2(self):
        """Test projecting StateSE2 onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = PoseSE2(1.0, 1.0, 0.0)
        distance = polyline.project(state)
        assert distance == 1.0

    def test_project_shapely_point(self):
        """Test projecting a shapely Point onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        point = geom.Point(1.0, 0.5)
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_project_ndarray(self):
        """Test projecting a raw ndarray onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        point = np.array([1.0, 0.0])
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_from_array_invalid_last_dim(self):
        """Test from_array with invalid last dimension raises ValueError."""
        array = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid polyline array shape"):
            PolylineSE2.from_array(array)

    def test_subline_returns_polyline_se2(self):
        """subline returns a PolylineSE2 instance."""
        polyline = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert isinstance(sub, PolylineSE2)

    def test_subline_full_range(self):
        """subline(0, length) reproduces the original SE2 array."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        sub = polyline.subline(0.0, polyline.length)
        np.testing.assert_array_almost_equal(sub.array, array)
        assert sub.length == pytest.approx(polyline.length)

    def test_subline_partial_range(self):
        """subline cuts to a partial range with the expected length."""
        polyline = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert sub.length == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(sub.array, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    def test_subline_exact_endpoints(self):
        """subline endpoints match interpolate(start) / interpolate(end)."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, np.pi / 4], [4.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        start_d, end_d = 1.0, 3.0
        sub = polyline.subline(start_d, end_d)
        # XY must match exactly; yaw can differ by 2*pi due to unwrap re-application.
        np.testing.assert_array_almost_equal(sub.array[0, :2], polyline.interpolate(start_d).array[:2])
        np.testing.assert_array_almost_equal(sub.array[-1, :2], polyline.interpolate(end_d).array[:2])

    def test_subline_normalized(self):
        """Normalized distances produce the same result as absolute distances."""
        polyline = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub_abs = polyline.subline(1.0, 3.0)
        sub_norm = polyline.subline(0.25, 0.75, normalized=True)
        np.testing.assert_array_almost_equal(sub_abs.array, sub_norm.array)

    def test_subline_clips_outside_range(self):
        """Distances outside [0, length] are clipped to the polyline bounds."""
        array = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        sub = polyline.subline(-5.0, polyline.length + 5.0)
        np.testing.assert_array_almost_equal(sub.array, array)

    def test_subline_swaps_reversed(self):
        """Reversed (start > end) is silently swapped."""
        polyline = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        np.testing.assert_array_almost_equal(
            polyline.subline(3.0, 1.0).array,
            polyline.subline(1.0, 3.0).array,
        )

    def test_subline_raises_on_zero_length(self):
        """subline raises ValueError when start_distance == end_distance after clipping."""
        polyline = PolylineSE2.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        with pytest.raises(ValueError, match="start_distance != end_distance"):
            polyline.subline(2.0, 2.0)

    def test_subline_preserves_intermediate_vertices(self):
        """Vertices strictly between start and end appear in the result."""
        array = np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        sub = polyline.subline(0.5, 3.5)
        assert sub.array.shape == (5, 3)
        np.testing.assert_array_almost_equal(sub.array[:, 0], [0.5, 1.0, 2.0, 3.0, 3.5])

    def test_subline_yaw_at_endpoints(self):
        """Interpolated yaw at endpoints matches interpolate(d).yaw modulo 2*pi."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, np.pi / 4], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        start_d = polyline.length / 4
        end_d = 3 * polyline.length / 4
        sub = polyline.subline(start_d, end_d)
        expected_start_yaw = polyline.interpolate(start_d).yaw
        expected_end_yaw = polyline.interpolate(end_d).yaw
        # PolylineSE2.__init__ re-unwraps yaws, so the stored yaw may differ from the
        # normalized [-pi, pi] value by a multiple of 2*pi. Compare modulo 2*pi.
        assert np.isclose((sub.array[0, 2] - expected_start_yaw) % (2 * np.pi), 0.0, atol=1e-9) or np.isclose(
            (sub.array[0, 2] - expected_start_yaw) % (2 * np.pi), 2 * np.pi, atol=1e-9
        )
        assert np.isclose((sub.array[-1, 2] - expected_end_yaw) % (2 * np.pi), 0.0, atol=1e-9) or np.isclose(
            (sub.array[-1, 2] - expected_end_yaw) % (2 * np.pi), 2 * np.pi, atol=1e-9
        )


class TestPolyline3D:
    """Test class for Polyline3D."""

    def test_from_linestring_with_z(self):
        """Test creating Polyline3D from LineString with Z coordinates."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        assert isinstance(polyline, Polyline3D)
        assert polyline.linestring.has_z

    def test_from_linestring_without_z(self):
        """Test creating Polyline3D from LineString without Z coordinates."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        assert isinstance(polyline, Polyline3D)
        assert polyline.linestring.has_z

    def test_from_array(self):
        """Test creating Polyline3D from 3D array."""
        array = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        assert isinstance(polyline, Polyline3D)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_invalid_shape(self):
        """Test creating Polyline3D from invalid array shape."""
        array = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            Polyline3D.from_array(array)

        array = np.array([[0.0], [1.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            Polyline3D.from_array(array)

    def test_array_property(self):
        """Test array property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        array = polyline.array
        assert array.shape == (3, 3)
        assert array.dtype == np.float64
        np.testing.assert_array_almost_equal(array, coords)

    def test_polyline_2d_property(self):
        """Test polyline_2d property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        polyline_2d = polyline.polyline_2d
        assert isinstance(polyline_2d, Polyline2D)
        assert not polyline_2d.linestring.has_z

    def test_polyline_se2_property(self):
        """Test polyline_se2 property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        polyline_se2 = polyline.polyline_se2
        assert isinstance(polyline_se2, PolylineSE2)

    def test_length_property(self):
        """Test length property."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        assert polyline.length == 2.0

        coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        assert polyline.length == 2.0

        coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        assert polyline.length == 2 * np.sqrt(3)

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = polyline.interpolate(np.sqrt(2))
        assert isinstance(point, Point3D)
        assert point.x == 1.0
        assert point.y == 0.0
        assert point.z == 1.0

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        points = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = polyline.interpolate(0.5, normalized=True)
        assert isinstance(point, Point3D)
        assert point.x == 1.0
        assert point.y == 0.0
        assert point.z == 1.0

    def test_project_point2d(self):
        """Test projecting Point2D onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        assert distance == 1.0

    def test_project_point3d(self):
        """Test projecting Point3D onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = Point3D(1.0, 1.0, 1.0)
        distance = polyline.project(point)
        assert distance == 1.0

    def test_project_shapely_point(self):
        """Test projecting a shapely Point onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = geom.Point(1.0, 0.5, 0.0)
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_project_ndarray(self):
        """Test projecting a raw ndarray onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = np.array([1.0, 0.0, 0.0])
        distance = polyline.project(point)
        assert distance == pytest.approx(1.0)

    def test_from_array_2d_input(self):
        """Test creating Polyline3D from 2D array (N, 2)."""
        array = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        assert isinstance(polyline, Polyline3D)
        # Z should be zero-padded with DEFAULT_Z
        assert polyline.array.shape == (3, 3)

    def test_project_pose_se2(self):
        """Test projecting PoseSE2 onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        pose = PoseSE2(1.0, 0.5, 0.0)
        distance = polyline.project(pose)
        assert distance == pytest.approx(1.0)

    def test_subline_returns_polyline3d(self):
        """subline returns a Polyline3D instance."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert isinstance(sub, Polyline3D)

    def test_subline_full_range(self):
        """subline(0, length) reproduces the original polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [4.0, 0.0, 2.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        sub = polyline.subline(0.0, polyline.length)
        np.testing.assert_array_almost_equal(sub.array, array)
        assert sub.length == pytest.approx(polyline.length)

    def test_subline_partial_range(self):
        """subline cuts to a partial range with the expected length."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub = polyline.subline(1.0, 3.0)
        assert sub.length == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(sub.array, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    def test_subline_exact_endpoints(self):
        """subline endpoints equal interpolate(start) / interpolate(end)."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 4.0]], dtype=np.float64))
        start_d, end_d = 1.0, polyline.length - 1.0
        sub = polyline.subline(start_d, end_d)
        np.testing.assert_array_almost_equal(sub.array[0], polyline.interpolate(start_d).array)
        np.testing.assert_array_almost_equal(sub.array[-1], polyline.interpolate(end_d).array)

    def test_subline_normalized(self):
        """Normalized distances produce the same result as absolute distances."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        sub_abs = polyline.subline(1.0, 3.0)
        sub_norm = polyline.subline(0.25, 0.75, normalized=True)
        np.testing.assert_array_almost_equal(sub_abs.array, sub_norm.array)

    def test_subline_clips_outside_range(self):
        """Distances outside [0, length] are clipped to the polyline bounds."""
        array = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        sub = polyline.subline(-5.0, polyline.length + 5.0)
        np.testing.assert_array_almost_equal(sub.array, array)

    def test_subline_swaps_reversed(self):
        """Reversed (start > end) is silently swapped."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        np.testing.assert_array_almost_equal(
            polyline.subline(3.0, 1.0).array,
            polyline.subline(1.0, 3.0).array,
        )

    def test_subline_raises_on_zero_length(self):
        """subline raises ValueError when start_distance == end_distance after clipping."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64))
        with pytest.raises(ValueError, match="start_distance != end_distance"):
            polyline.subline(2.0, 2.0)

    def test_subline_preserves_intermediate_vertices(self):
        """Vertices strictly between start and end appear in the result."""
        array = np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        sub = polyline.subline(0.5, 3.5)
        assert sub.array.shape == (5, 3)
        np.testing.assert_array_almost_equal(sub.array[:, 0], [0.5, 1.0, 2.0, 3.0, 3.5])

    def test_subline_preserves_z(self):
        """Z coordinates are preserved through subline (not silently dropped)."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        # Pick endpoint distances that land on the vertices to avoid floating-point drift.
        sub = polyline.subline(np.sqrt(2), 2 * np.sqrt(2))
        # Expected: start vertex (1, 0, 1), end vertex (2, 0, 2)
        np.testing.assert_array_almost_equal(sub.array, [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
