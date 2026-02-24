import numpy as np
import pytest
from shapely.geometry import LineString

from py123d.geometry.utils.polyline_utils import (
    get_linestring_yaws,
    get_path_progress_2d,
    get_path_progress_3d,
    get_points_2d_yaws,
    offset_points_perpendicular,
)


class TestGetLinestringYaws:
    """Tests for get_linestring_yaws."""

    def test_horizontal_linestring(self):
        """Test yaw computation on a horizontal linestring (heading = 0)."""
        ls = LineString([(0, 0), (1, 0), (2, 0)])
        yaws = get_linestring_yaws(ls)
        np.testing.assert_allclose(yaws, [0.0, 0.0, 0.0], atol=1e-10)

    def test_vertical_linestring(self):
        """Test yaw computation on a vertical linestring (heading = pi/2)."""
        ls = LineString([(0, 0), (0, 1), (0, 2)])
        yaws = get_linestring_yaws(ls)
        np.testing.assert_allclose(yaws, [np.pi / 2, np.pi / 2, np.pi / 2], atol=1e-10)

    def test_diagonal_linestring(self):
        """Test yaw computation on a 45-degree diagonal linestring."""
        ls = LineString([(0, 0), (1, 1), (2, 2)])
        yaws = get_linestring_yaws(ls)
        expected = np.pi / 4
        np.testing.assert_allclose(yaws, [expected, expected, expected], atol=1e-10)

    def test_last_yaw_matches_second_last(self):
        """Test that the last yaw is duplicated from the second-to-last."""
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        yaws = get_linestring_yaws(ls)
        assert yaws[-1] == yaws[-2]


class TestGetPoints2DYaws:
    """Tests for get_points_2d_yaws."""

    def test_basic(self):
        """Test yaw computation on basic points."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        yaws = get_points_2d_yaws(points)
        np.testing.assert_allclose(yaws, [0.0, 0.0, 0.0], atol=1e-10)

    def test_length_consistency(self):
        """Test that yaw output length matches the number of input points."""
        points = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 2.0], [5.0, 0.0]])
        yaws = get_points_2d_yaws(points)
        assert len(yaws) == len(points)


class TestGetPathProgress2D:
    """Tests for get_path_progress_2d."""

    def test_progress_with_point2d_array(self):
        """Test cumulative path progress for 2D points."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        progress = get_path_progress_2d(points)
        assert progress[0] == 0.0
        # Since we're computing based on diffs, the progress should be cumulative distances.
        # Points go along the x-axis: 0→1→2, distances are 1, 1
        assert len(progress) == 3

    def test_progress_with_pose_se2_array(self):
        """Test cumulative path progress for SE2 poses."""
        poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        progress = get_path_progress_2d(poses)
        assert progress[0] == 0.0
        assert len(progress) == 3
        np.testing.assert_allclose(progress, [0.0, 1.0, 2.0], atol=1e-10)

    def test_progress_invalid_shape_raises(self):
        """Test that invalid array shape raises ValueError."""
        invalid = np.array([[0.0, 0.0, 0.0, 0.0]])  # shape (1, 4) - neither 2 nor 3
        with pytest.raises(ValueError, match="Invalid points_array shape"):
            get_path_progress_2d(invalid)

    def test_progress_2d_point2d_xy_diff_correctness(self):
        """Test that path progress correctly uses both X and Y diffs for 2D points.

        This test verifies the path progress computation accounts for both X and Y coordinates.
        A path with movement only in the Y direction should still accumulate distance.
        """
        # Path that moves only in Y direction: distances should be 1.0 each step
        points_y_only = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        progress = get_path_progress_2d(points_y_only)
        np.testing.assert_allclose(progress, [0.0, 1.0, 2.0], atol=1e-10)


class TestGetPathProgress3D:
    """Tests for get_path_progress_3d."""

    def test_progress_basic(self):
        """Test cumulative path progress for 3D points along x-axis."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        progress = get_path_progress_3d(points)
        np.testing.assert_allclose(progress, [0.0, 1.0, 2.0], atol=1e-10)

    def test_progress_diagonal(self):
        """Test cumulative path progress on a diagonal path."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        progress = get_path_progress_3d(points)
        np.testing.assert_allclose(progress, [0.0, np.sqrt(3.0)], atol=1e-10)

    def test_progress_invalid_shape_raises(self):
        """Test that invalid array shape raises ValueError."""
        invalid = np.array([[0.0, 0.0]])  # shape (1, 2) - not 3
        with pytest.raises(ValueError, match="Invalid points_array shape"):
            get_path_progress_3d(invalid)


class TestOffsetPointsPerpendicular:
    """Tests for offset_points_perpendicular."""

    def test_offset_point2d_array(self):
        """Test perpendicular offset of 2D points."""
        # Straight horizontal line: offset should move points in Y direction
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        offset = 1.0
        result = offset_points_perpendicular(points, offset)
        # Yaw for horizontal line is 0, so perpendicular (left) is +Y
        np.testing.assert_allclose(result[:, 1], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(result[:, 0], [0.0, 1.0, 2.0], atol=1e-10)

    def test_offset_pose_se2_array(self):
        """Test perpendicular offset of SE2 poses."""
        # SE2 poses with yaw=0 (horizontal), offset in Y
        poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        offset = 1.0
        result = offset_points_perpendicular(poses, offset)
        np.testing.assert_allclose(result[:, 1], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(result[:, 0], [0.0, 1.0, 2.0], atol=1e-10)

    def test_offset_invalid_shape_raises(self):
        """Test that invalid array shape raises ValueError."""
        invalid = np.array([[0.0, 0.0, 0.0, 0.0]])  # shape (1, 4)
        with pytest.raises(ValueError, match="Invalid points_array shape"):
            offset_points_perpendicular(invalid, 1.0)

    def test_offset_zero(self):
        """Test that zero offset returns the original points."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = offset_points_perpendicular(points, 0.0)
        np.testing.assert_allclose(result, points, atol=1e-10)
