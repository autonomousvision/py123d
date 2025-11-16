import unittest

import numpy as np
import pytest
import shapely.geometry as geom

from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, Point2D, Point3D, PoseSE2, PoseSE3
from py123d.geometry.geometry_index import (
    BoundingBoxSE2Index,
    BoundingBoxSE3Index,
    Corners2DIndex,
    Corners3DIndex,
    Point2DIndex,
)


class TestBoundingBoxSE2:
    """Unit tests for BoundingBoxSE2 class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.center = PoseSE2(1.0, 2.0, 0.5)
        self.length = 4.0
        self.width = 2.0
        self.bbox = BoundingBoxSE2(self.center, self.length, self.width)

    def test_init(self):
        """Test BoundingBoxSE2 initialization."""
        bbox = BoundingBoxSE2(self.center, self.length, self.width)
        assert bbox.length == self.length
        assert bbox.width == self.width
        np.testing.assert_array_equal(bbox.center_se2.array, self.center.array)

    def test_from_array(self):
        """Test BoundingBoxSE2.from_array method."""
        array = np.array([1.0, 2.0, 0.5, 4.0, 2.0])
        bbox = BoundingBoxSE2.from_array(array)
        np.testing.assert_array_equal(bbox.array, array)

    def test_from_array_copy(self):
        """Test BoundingBoxSE2.from_array with copy parameter."""
        array = np.array([1.0, 2.0, 0.5, 4.0, 2.0])
        bbox_copy = BoundingBoxSE2.from_array(array, copy=True)
        bbox_no_copy = BoundingBoxSE2.from_array(array, copy=False)

        array[0] = 999.0
        assert bbox_copy.array[0] != 999.0
        assert bbox_no_copy.array[0] == 999.0

    def test_properties(self):
        """Test BoundingBoxSE2 properties."""
        assert self.bbox.length == self.length
        assert self.bbox.width == self.width
        np.testing.assert_array_equal(self.bbox.center_se2.array, self.center.array)

    def test_array_property(self):
        """Test array property."""
        expected = np.array([1.0, 2.0, 0.5, 4.0, 2.0])
        np.testing.assert_array_equal(self.bbox.array, expected)

    def test_array_mixin(self):
        """Test that BoundingBoxSE2 is an instance of ArrayMixin."""
        assert isinstance(self.bbox, ArrayMixin)

        expected = np.array([1.0, 2.0, 0.5, 4.0, 2.0], dtype=np.float16)
        output_array = np.array(self.bbox, dtype=np.float16)
        np.testing.assert_array_equal(output_array, expected)
        assert output_array.dtype == np.float16
        assert output_array.shape == (len(BoundingBoxSE2Index),)

    def test_bounding_box_se2_property(self):
        """Test bounding_box_se2 property returns self."""
        assert self.bbox.bounding_box_se2 is self.bbox

    def test_corners_array(self):
        """Test corners_array property."""
        corners = self.bbox.corners_array
        assert corners.shape == (len(Corners2DIndex), len(Point2DIndex))
        assert isinstance(corners, np.ndarray)

    def test_corners_dict(self):
        """Test corners_dict property."""
        corners_dict = self.bbox.corners_dict
        assert len(corners_dict) == len(Corners2DIndex)
        for index in Corners2DIndex:
            assert index in corners_dict
            assert isinstance(corners_dict[index], Point2D)

    def test_shapely_polygon(self):
        """Test shapely_polygon property."""
        polygon = self.bbox.shapely_polygon
        assert isinstance(polygon, geom.Polygon)
        assert polygon.area == pytest.approx(self.length * self.width)

    def test_array_assertions(self):
        """Test array assertions in from_array."""
        # Test 2D array
        with pytest.raises(AssertionError):
            BoundingBoxSE2.from_array(np.array([[1, 2, 3, 4, 5]]))

        # Test wrong size
        with pytest.raises(AssertionError):
            BoundingBoxSE2.from_array(np.array([1, 2, 3, 4]))


class TestBoundingBoxSE3:
    """Unit tests for BoundingBoxSE3 class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.array = np.array([1.0, 2.0, 3.0, 0.98185617, 0.06407135, 0.09115755, 0.1534393, 4.0, 2.0, 1.5])
        self.center_se3 = PoseSE3(1.0, 2.0, 3.0, 0.98185617, 0.06407135, 0.09115755, 0.1534393)
        self.length = 4.0
        self.width = 2.0
        self.height = 1.5
        self.bbox = BoundingBoxSE3(self.center_se3, self.length, self.width, self.height)

    def test_init(self):
        """Test BoundingBoxSE3 initialization."""
        bbox = BoundingBoxSE3(self.center_se3, self.length, self.width, self.height)
        assert bbox.length == self.length
        assert bbox.width == self.width
        assert bbox.height == self.height
        np.testing.assert_array_equal(bbox.center_se3.array, self.center_se3.array)

    def test_from_array(self):
        """Test BoundingBoxSE3.from_array method."""
        array = self.array.copy()
        bbox = BoundingBoxSE3.from_array(array)
        np.testing.assert_array_equal(bbox.array, array)

    def test_from_array_copy(self):
        """Test BoundingBoxSE3.from_array with copy parameter."""
        array = self.array.copy()
        bbox_copy = BoundingBoxSE3.from_array(array, copy=True)
        bbox_no_copy = BoundingBoxSE3.from_array(array, copy=False)

        array[0] = 999.0
        assert bbox_copy.array[0] != 999.0
        assert bbox_no_copy.array[0] == 999.0

    def test_properties(self):
        """Test BoundingBoxSE3 properties."""
        assert self.bbox.length == self.length
        assert self.bbox.width == self.width
        assert self.bbox.height == self.height
        np.testing.assert_array_equal(self.bbox.center_se3.array, self.center_se3.array)

    def test_array_property(self):
        """Test array property."""
        expected = self.array.copy()
        np.testing.assert_array_equal(self.bbox.array, expected)

    def test_array_mixin(self):
        """Test that BoundingBoxSE3 is an instance of ArrayMixin."""
        assert isinstance(self.bbox, ArrayMixin)

        expected = np.array(self.array, dtype=np.float16)
        output_array = np.array(self.bbox, dtype=np.float16)
        np.testing.assert_array_equal(output_array, expected)
        assert output_array.dtype == np.float16
        assert output_array.shape == (len(BoundingBoxSE3Index),)

    def test_bounding_box_se2_property(self):
        """Test bounding_box_se2 property."""
        bbox_2d = self.bbox.bounding_box_se2
        assert isinstance(bbox_2d, BoundingBoxSE2)
        assert bbox_2d.length == self.length
        assert bbox_2d.width == self.width
        assert bbox_2d.center_se2.x == self.center_se3.x
        assert bbox_2d.center_se2.y == self.center_se3.y
        assert bbox_2d.center_se2.yaw == self.center_se3.euler_angles.yaw

    def test_corners_array(self):
        """Test corners_array property."""
        corners = self.bbox.corners_array
        assert corners.shape == (8, 3)
        assert isinstance(corners, np.ndarray)

    def test_corners_dict(self):
        """Test corners_dict property."""
        corners_dict = self.bbox.corners_dict
        assert len(corners_dict) == 8
        for index in Corners3DIndex:
            assert index in corners_dict
            assert isinstance(corners_dict[index], Point3D)

    def test_shapely_polygon(self):
        """Test shapely_polygon property."""
        polygon = self.bbox.shapely_polygon
        assert isinstance(polygon, geom.Polygon)
        assert polygon.area == pytest.approx(self.length * self.width)

    def test_array_assertions(self):
        """Test array assertions in from_array."""
        # Test 2D array
        with pytest.raises(AssertionError):
            BoundingBoxSE3.from_array(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))

        # Test wrong size, less than required
        with pytest.raises(AssertionError):
            BoundingBoxSE3.from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

        # Test wrong size, greater than required
        with pytest.raises(AssertionError):
            BoundingBoxSE3.from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

    def test_zero_dimensions(self):
        """Test bounding box with zero dimensions."""
        center = PoseSE3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bbox = BoundingBoxSE3(center, 0.0, 0.0, 0.0)
        assert bbox.length == 0.0
        assert bbox.width == 0.0
        assert bbox.height == 0.0


if __name__ == "__main__":
    unittest.main()
