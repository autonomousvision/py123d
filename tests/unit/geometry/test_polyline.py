import unittest

import numpy as np
import shapely.geometry as geom

from py123d.geometry import Point2D, Point3D, Polyline2D, Polyline3D, PolylineSE2, PoseSE2


class TestPolyline2D(unittest.TestCase):
    """Test class for Polyline2D."""

    def test_from_linestring(self):
        """Test creating Polyline2D from LineString."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        self.assertIsInstance(polyline, Polyline2D)
        self.assertTrue(polyline.linestring.equals(linestring))

    def test_from_linestring_with_z(self):
        """Test creating Polyline2D from LineString with Z coordinates."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        self.assertIsInstance(polyline, Polyline2D)
        self.assertFalse(polyline.linestring.has_z)

    def test_from_array_2d(self):
        """Test creating Polyline2D from 2D array."""
        array = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        polyline = Polyline2D.from_array(array)
        self.assertIsInstance(polyline, Polyline2D)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_3d(self):
        """Test creating Polyline2D from 3D array."""
        array = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float32)
        polyline = Polyline2D.from_array(array)
        self.assertIsInstance(polyline, Polyline2D)
        expected = array[:, :2]
        np.testing.assert_array_almost_equal(polyline.array, expected)

    def test_from_array_invalid_shape(self):
        """Test creating Polyline2D from invalid array shape."""
        array = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            Polyline2D.from_array(array)

    def test_array_property(self):
        """Test array property."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        array = polyline.array
        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(array.dtype, np.float64)
        np.testing.assert_array_almost_equal(array, coords)

    def test_length_property(self):
        """Test length property."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        self.assertEqual(polyline.length, 2.0)

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = polyline.interpolate(1.0)
        self.assertIsInstance(point, Point2D)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 0.0)

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        points = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape, (3, 2))
        expected = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_array_almost_equal(points, expected)

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = polyline.interpolate(0.5, normalized=True)
        self.assertIsInstance(point, Point2D)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 0.0)

    def test_project_point2d(self):
        """Test projecting Point2D onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        self.assertEqual(distance, 1.0)

    def test_project_statese2(self):
        """Test projecting StateSE2 onto polyline."""
        coords = [(0.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        state = PoseSE2(1.0, 1.0, 0.0)
        distance = polyline.project(state)
        self.assertEqual(distance, 1.0)

    def test_polyline_se2_property(self):
        """Test polyline_se2 property."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline2D.from_linestring(linestring)
        polyline_se2 = polyline.polyline_se2
        self.assertIsInstance(polyline_se2, PolylineSE2)


class TestPolylineSE2(unittest.TestCase):
    """Test class for PolylineSE2."""

    def test_from_linestring(self):
        """Test creating PolylineSE2 from LineString."""
        coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = PolylineSE2.from_linestring(linestring)
        self.assertIsInstance(polyline, PolylineSE2)
        self.assertEqual(polyline.array.shape, (3, 3))

    def test_from_array_2d(self):
        """Test creating PolylineSE2 from 2D array."""
        array = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        polyline = PolylineSE2.from_array(array)
        self.assertIsInstance(polyline, PolylineSE2)
        self.assertEqual(polyline.array.shape, (3, 3))

    def test_from_array_se2(self):
        """Test creating PolylineSE2 from SE2 array."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        polyline = PolylineSE2.from_array(array)
        self.assertIsInstance(polyline, PolylineSE2)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_invalid_shape(self):
        """Test creating PolylineSE2 from invalid array shape."""
        array = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            PolylineSE2.from_array(array)

    def test_from_discrete_se2(self):
        """Test creating PolylineSE2 from discrete SE2 states."""
        states = [PoseSE2(0.0, 0.0, 0.0), PoseSE2(1.0, 0.0, 0.0), PoseSE2(2.0, 0.0, 0.0)]
        polyline = PolylineSE2.from_discrete_se2(states)
        self.assertIsInstance(polyline, PolylineSE2)
        self.assertEqual(polyline.array.shape, (3, 3))

    def test_length_property(self):
        """Test length property."""
        array = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        self.assertEqual(polyline.length, 2.0)

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = polyline.interpolate(1.0)
        self.assertIsInstance(state, PoseSE2)
        self.assertEqual(state.x, 1.0)
        self.assertEqual(state.y, 0.0)

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        states = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(states, np.ndarray)
        self.assertEqual(states.shape, (3, 3))

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = polyline.interpolate(0.5, normalized=True)
        self.assertIsInstance(state, PoseSE2)
        self.assertEqual(state.x, 1.0)
        self.assertEqual(state.y, 0.0)

    def test_project_point2d(self):
        """Test projecting Point2D onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        self.assertEqual(distance, 1.0)

    def test_project_statese2(self):
        """Test projecting StateSE2 onto SE2 polyline."""
        array = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        polyline = PolylineSE2.from_array(array)
        state = PoseSE2(1.0, 1.0, 0.0)
        distance = polyline.project(state)
        self.assertEqual(distance, 1.0)


class TestPolyline3D(unittest.TestCase):
    """Test class for Polyline3D."""

    def test_from_linestring_with_z(self):
        """Test creating Polyline3D from LineString with Z coordinates."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        self.assertIsInstance(polyline, Polyline3D)
        self.assertTrue(polyline.linestring.has_z)

    def test_from_linestring_without_z(self):
        """Test creating Polyline3D from LineString without Z coordinates."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        self.assertIsInstance(polyline, Polyline3D)
        self.assertTrue(polyline.linestring.has_z)

    def test_from_array(self):
        """Test creating Polyline3D from 3D array."""
        array = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float64)
        polyline = Polyline3D.from_array(array)
        self.assertIsInstance(polyline, Polyline3D)
        np.testing.assert_array_almost_equal(polyline.array, array)

    def test_from_array_invalid_shape(self):
        """Test creating Polyline3D from invalid array shape."""
        array = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Polyline3D.from_array(array)

    def test_array_property(self):
        """Test array property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        array = polyline.array
        self.assertEqual(array.shape, (3, 3))
        self.assertEqual(array.dtype, np.float64)
        np.testing.assert_array_almost_equal(array, coords)

    def test_polyline_2d_property(self):
        """Test polyline_2d property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        polyline_2d = polyline.polyline_2d
        self.assertIsInstance(polyline_2d, Polyline2D)
        self.assertFalse(polyline_2d.linestring.has_z)

    def test_polyline_se2_property(self):
        """Test polyline_se2 property."""
        coords = [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        polyline_se2 = polyline.polyline_se2
        self.assertIsInstance(polyline_se2, PolylineSE2)

    def test_length_property(self):
        """Test length property."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        self.assertEqual(polyline.length, 2.0)

    def test_interpolate_single_distance(self):
        """Test interpolation with single distance."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = polyline.interpolate(1.0)
        self.assertIsInstance(point, Point3D)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 0.0)
        self.assertEqual(point.z, 1.0)

    def test_interpolate_multiple_distances(self):
        """Test interpolation with multiple distances."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        points = polyline.interpolate(np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape, (3, 3))

    def test_interpolate_normalized(self):
        """Test normalized interpolation."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = polyline.interpolate(0.5, normalized=True)
        self.assertIsInstance(point, Point3D)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 0.0)
        self.assertEqual(point.z, 1.0)

    def test_project_point2d(self):
        """Test projecting Point2D onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = Point2D(1.0, 1.0)
        distance = polyline.project(point)
        self.assertEqual(distance, 1.0)

    def test_project_point3d(self):
        """Test projecting Point3D onto 3D polyline."""
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        linestring = geom.LineString(coords)
        polyline = Polyline3D.from_linestring(linestring)
        point = Point3D(1.0, 1.0, 1.0)
        distance = polyline.project(point)
        self.assertEqual(distance, 1.0)
