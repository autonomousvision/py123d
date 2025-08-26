import unittest

import numpy as np

from d123.geometry import Vector2D, Vector2DIndex, Vector3D, Vector3DIndex


class TestVector2D(unittest.TestCase):
    """Unit tests for Vector2D class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x_coord = 3.5
        self.y_coord = 4.2
        self.vector = Vector2D(x=self.x_coord, y=self.y_coord)
        self.test_array = np.zeros([2], dtype=np.float64)
        self.test_array[Vector2DIndex.X] = self.x_coord
        self.test_array[Vector2DIndex.Y] = self.y_coord

    def test_init(self):
        """Test Vector2D initialization."""
        vector = Vector2D(1.0, 2.0)
        self.assertEqual(vector.x, 1.0)
        self.assertEqual(vector.y, 2.0)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        vector = Vector2D.from_array(self.test_array)
        self.assertEqual(vector.x, self.x_coord)
        self.assertEqual(vector.y, self.y_coord)

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector2D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector2D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        array_wrong_length = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector2D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector2D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.vector.array, expected_array)
        self.assertEqual(self.vector.array.dtype, np.float64)
        self.assertEqual(self.vector.array.shape, (2,))

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float32)
        output_array = np.array(self.vector, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        self.assertEqual(output_array.dtype, np.float32)
        self.assertEqual(output_array.shape, (2,))

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.vector)
        self.assertEqual(coords, [self.x_coord, self.y_coord])

        # Test that it's actually iterable
        x, y = self.vector
        self.assertEqual(x, self.x_coord)
        self.assertEqual(y, self.y_coord)

    def test_hash(self):
        """Test the __hash__ method."""
        vector_dict = {self.vector: "test"}
        self.assertIn(self.vector, vector_dict)
        self.assertEqual(vector_dict[self.vector], "test")


class TestVector3D(unittest.TestCase):
    """Unit tests for Vector3D class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x_coord = 3.5
        self.y_coord = 4.2
        self.z_coord = 5.1
        self.vector = Vector3D(self.x_coord, self.y_coord, self.z_coord)
        self.test_array = np.zeros((3,), dtype=np.float64)
        self.test_array[Vector3DIndex.X] = self.x_coord
        self.test_array[Vector3DIndex.Y] = self.y_coord
        self.test_array[Vector3DIndex.Z] = self.z_coord

    def test_init(self):
        """Test Vector3D initialization."""
        vector = Vector3D(1.0, 2.0, 3.0)
        self.assertEqual(vector.x, 1.0)
        self.assertEqual(vector.y, 2.0)
        self.assertEqual(vector.z, 3.0)

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        vector = Vector3D.from_array(self.test_array)
        self.assertEqual(vector.x, self.x_coord)
        self.assertEqual(vector.y, self.y_coord)
        self.assertEqual(vector.z, self.z_coord)

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector3D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector3D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        array_wrong_length = np.array([1.0, 2.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector3D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with self.assertRaises(AssertionError):
            Vector3D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.vector.array, expected_array)
        self.assertEqual(self.vector.array.dtype, np.float64)
        self.assertEqual(self.vector.array.shape, (3,))

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float32)
        output_array = np.array(self.vector, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        self.assertEqual(output_array.dtype, np.float32)
        self.assertEqual(output_array.shape, (3,))

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.vector)
        self.assertEqual(coords, [self.x_coord, self.y_coord, self.z_coord])

        # Test that it's actually iterable
        x, y, z = self.vector
        self.assertEqual(x, self.x_coord)
        self.assertEqual(y, self.y_coord)
        self.assertEqual(z, self.z_coord)

    def test_hash(self):
        """Test the __hash__ method."""
        vector_dict = {self.vector: "test"}
        self.assertIn(self.vector, vector_dict)
        self.assertEqual(vector_dict[self.vector], "test")


if __name__ == "__main__":
    unittest.main()
