import numpy as np
import pytest

from py123d.geometry import Vector2D, Vector2DIndex, Vector3D, Vector3DIndex


class TestVector2D:
    """Unit tests for Vector2D class."""

    def setup_method(self):
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
        assert vector.x == 1.0
        assert vector.y == 2.0

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        vector = Vector2D.from_array(self.test_array)
        assert vector.x == self.x_coord
        assert vector.y == self.y_coord

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector2D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector2D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        array_wrong_length = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector2D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector2D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.vector.array, expected_array)
        assert self.vector.array.dtype == np.float64
        assert self.vector.array.shape == (2,)

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord], dtype=np.float32)
        output_array = np.array(self.vector, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        assert output_array.dtype == np.float32
        assert output_array.shape == (2,)

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.vector)
        assert coords == [self.x_coord, self.y_coord]

        # Test that it's actually iterable
        x, y = self.vector
        assert x == self.x_coord
        assert y == self.y_coord

    def test_magnitude(self):
        """Test magnitude computation."""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude == pytest.approx(5.0)

    def test_vector_2d_property(self):
        """Test that vector_2d returns self."""
        v = Vector2D(1.0, 2.0)
        assert v.vector_2d is v

    def test_add(self):
        """Test vector addition."""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        result = v1 + v2
        assert isinstance(result, Vector2D)
        assert result.x == 4.0
        assert result.y == 6.0

    def test_sub(self):
        """Test vector subtraction."""
        v1 = Vector2D(5.0, 7.0)
        v2 = Vector2D(1.0, 3.0)
        result = v1 - v2
        assert isinstance(result, Vector2D)
        assert result.x == 4.0
        assert result.y == 4.0

    def test_mul(self):
        """Test scalar multiplication."""
        v = Vector2D(2.0, 3.0)
        result = v * 2.5
        assert isinstance(result, Vector2D)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(7.5)

    def test_truediv(self):
        """Test scalar division."""
        v = Vector2D(6.0, 9.0)
        result = v / 3.0
        assert isinstance(result, Vector2D)
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(3.0)

    def test_neg(self):
        """Test negation of Vector2D."""
        v = Vector2D(3.0, -4.0)
        neg_v = -v
        assert isinstance(neg_v, Vector2D)
        assert neg_v.x == -3.0
        assert neg_v.y == 4.0

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        v = Vector2D(1.0, 2.0)
        r = repr(v)
        assert "Vector2D" in r


class TestVector3D:
    """Unit tests for Vector3D class."""

    def setup_method(self):
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
        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0

    def test_from_array_valid(self):
        """Test from_array class method with valid input."""
        vector = Vector3D.from_array(self.test_array)
        assert vector.x == self.x_coord
        assert vector.y == self.y_coord
        assert vector.z == self.z_coord

    def test_from_array_invalid_dimensions(self):
        """Test from_array with invalid array dimensions."""
        # 2D array should raise assertion error
        array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector3D.from_array(array_2d)

        # 3D array should raise assertion error
        array_3d = np.array([[[1.0]]], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector3D.from_array(array_3d)

    def test_from_array_invalid_shape(self):
        """Test from_array with invalid array shape."""
        array_wrong_length = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector3D.from_array(array_wrong_length)

        # Empty array
        empty_array = np.array([], dtype=np.float64)
        with pytest.raises(AssertionError):
            Vector3D.from_array(empty_array)

    def test_array_property(self):
        """Test the array property."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float64)
        np.testing.assert_array_equal(self.vector.array, expected_array)
        assert self.vector.array.dtype == np.float64
        assert self.vector.array.shape == (3,)

    def test_array_like(self):
        """Test the __array__ behavior."""
        expected_array = np.array([self.x_coord, self.y_coord, self.z_coord], dtype=np.float32)
        output_array = np.array(self.vector, dtype=np.float32)
        np.testing.assert_array_equal(output_array, expected_array)
        assert output_array.dtype == np.float32
        assert output_array.shape == (3,)

    def test_iter(self):
        """Test the __iter__ method."""
        coords = list(self.vector)
        assert coords == [self.x_coord, self.y_coord, self.z_coord]

        # Test that it's actually iterable
        x, y, z = self.vector
        assert x == self.x_coord
        assert y == self.y_coord
        assert z == self.z_coord

    def test_magnitude(self):
        """Test magnitude computation."""
        v = Vector3D(1.0, 2.0, 2.0)
        assert v.magnitude == pytest.approx(3.0)

    def test_vector_3d_property(self):
        """Test that vector_3d returns self."""
        v = Vector3D(1.0, 2.0, 3.0)
        assert v.vector_3d is v

    def test_vector_2d_property(self):
        """Test XY projection as Vector2D."""
        v = Vector3D(1.0, 2.0, 3.0)
        v2d = v.vector_2d
        assert isinstance(v2d, Vector2D)
        assert v2d.x == 1.0
        assert v2d.y == 2.0

    def test_add(self):
        """Test vector addition."""
        v1 = Vector3D(1.0, 2.0, 3.0)
        v2 = Vector3D(4.0, 5.0, 6.0)
        result = v1 + v2
        assert isinstance(result, Vector3D)
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_sub(self):
        """Test vector subtraction."""
        v1 = Vector3D(5.0, 7.0, 9.0)
        v2 = Vector3D(1.0, 2.0, 3.0)
        result = v1 - v2
        assert isinstance(result, Vector3D)
        assert result.x == 4.0
        assert result.y == 5.0
        assert result.z == 6.0

    def test_mul(self):
        """Test scalar multiplication."""
        v = Vector3D(2.0, 3.0, 4.0)
        result = v * 2.5
        assert isinstance(result, Vector3D)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(7.5)
        assert result.z == pytest.approx(10.0)

    def test_truediv(self):
        """Test scalar division."""
        v = Vector3D(6.0, 9.0, 12.0)
        result = v / 3.0
        assert isinstance(result, Vector3D)
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(3.0)
        assert result.z == pytest.approx(4.0)

    def test_neg(self):
        """Test negation of Vector3D."""
        v = Vector3D(3.0, -4.0, 5.0)
        neg_v = -v
        assert isinstance(neg_v, Vector3D)
        assert neg_v.x == -3.0
        assert neg_v.y == 4.0
        assert neg_v.z == -5.0

    def test_repr(self):
        """Test __repr__ returns a string containing the class name."""
        v = Vector3D(1.0, 2.0, 3.0)
        r = repr(v)
        assert "Vector3D" in r
