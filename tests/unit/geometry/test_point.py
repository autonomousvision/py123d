import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from py123d.geometry import Point2D, Point3D
from py123d.geometry.geometry_index import Point2DIndex, Point3DIndex


class TestPoint2D(unittest.TestCase):
    """Unit tests for Point2D class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x_coord = 3.5
        self.y_coord = 4.2
        self.point = Point2D(x=self.x_coord, y=self.y_coord)
        self.test_array = np.zeros([2], dtype=np.float64)
        self.test_array[Point2DIndex.X] = self.x_coord
        self.test_array[Point2DIndex.Y] = self.y_coord

    def test_init(self):
        """Test Point2D initialization."""
        point = Point2D(1.0, 2.0)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        # Mock Point2DIndex enum values
        point = Point2D.from_array(self.test_array)
        self.assertEqual(point.x, self.x_coord)
        self.assertEqual(point.y, self.y_coord)

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point2D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point2D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""

        array_wrong_length = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point2D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point2D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.point.array, expected_array)
        self.assertEqual(self.point.array.dtype, np.float64)
        self.assertEqual(self.point.array.shape, (2,))

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float32)
        output_array = np.array(self.point, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        self.assertEqual(output_array.dtype, np.float32)
        self.assertEqual(output_array.shape, (2,))

    def test_shapely_point_property(self):
        """Test the shapely_point property."""
        with patch("shapely.geometry.Point") as mock_point:
            mock_point_instance = MagicMock()
            mock_point.return_value = mock_point_instance

            result = self.point.shapely_point

            mock_point.assert_called_once_with(self.x_coord, self.y_coord)
            self.assertEqual(result, mock_point_instance)

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.point)
        self.assertEqual(coords, [self.x_coord, self.y_coord])

        # Test that it's actually iterable
        x, y = self.point
        self.assertEqual(x, self.x_coord)
        self.assertEqual(y, self.y_coord)


class TestPoint3D(unittest.TestCase):
    """Unit tests for Point3D class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x_coord = 3.5
        self.y_coord = 4.2
        self.z_coord = 5.1
        self.point = Point3D(self.x_coord, self.y_coord, self.z_coord)
        self.test_array = np.zeros((3,), dtype=np.float64)
        self.test_array[Point3DIndex.X] = self.x_coord
        self.test_array[Point3DIndex.Y] = self.y_coord
        self.test_array[Point3DIndex.Z] = self.z_coord

    def test_init(self):
        """Test Point3D initialization."""
        point = Point3D(1.0, 2.0, 3.0)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)
        self.assertEqual(point.z, 3.0)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        # Mock Point3DIndex enum values
        point = Point3D.from_array(self.test_array)
        self.assertEqual(point.x, self.x_coord)
        self.assertEqual(point.y, self.y_coord)
        self.assertEqual(point.z, self.z_coord)

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point3D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point3D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""

        array_wrong_length = np.array([1.0, 2.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point3D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Point3D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.point.array, expected_array)
        self.assertEqual(self.point.array.dtype, np.float64)
        self.assertEqual(self.point.array.shape, (3,))

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float32)
        output_array = np.array(self.point, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        self.assertEqual(output_array.dtype, np.float32)
        self.assertEqual(output_array.shape, (3,))

    def test_shapely_point_property(self):
        """Test the shapely_point property."""
        with patch("shapely.geometry.Point") as mock_point:
            mock_point_instance = MagicMock()
            mock_point.return_value = mock_point_instance

            result = self.point.shapely_point

            mock_point.assert_called_once_with(self.x_coord, self.y_coord, self.z_coord)
            self.assertEqual(result, mock_point_instance)

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.point)
        self.assertEqual(coords, [self.x_coord, self.y_coord, self.z_coord])

        # Test that it's actually iterable
        x, y, z = self.point
        self.assertEqual(x, self.x_coord)
        self.assertEqual(y, self.y_coord)
        self.assertEqual(z, self.z_coord)


if __name__ == "__main__":
    unittest.main()
