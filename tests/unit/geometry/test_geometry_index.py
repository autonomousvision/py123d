import numpy as np

from py123d.geometry.geometry_index import (
    BoundingBoxSE3Index,
    Corners3DIndex,
    MatrixSE2Index,
    MatrixSE3Index,
    MatrixSO2Index,
    MatrixSO3Index,
    PoseSE2Index,
    PoseSE3Index,
    QuaternionIndex,
)


class TestPoseSE2Index:
    """Tests for PoseSE2Index classproperties."""

    def test_se2_slice(self):
        """Test that SE2 slice covers all x, y, yaw components."""
        array = np.array([1.0, 2.0, 0.5])
        result = array[PoseSE2Index.SE2]
        np.testing.assert_array_equal(result, array)


class TestQuaternionIndex:
    """Tests for QuaternionIndex classproperties."""

    def test_scalar_property(self):
        """Test that SCALAR indexes the qw component."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        assert q[QuaternionIndex.SCALAR] == 1.0

    def test_vector_property(self):
        """Test that VECTOR slice indexes the qx, qy, qz components."""
        q = np.array([1.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(q[QuaternionIndex.VECTOR], [0.1, 0.2, 0.3])


class TestPoseSE3Index:
    """Tests for PoseSE3Index classproperties."""

    def test_scalar_property(self):
        """Test that SCALAR slice indexes the qw component."""
        pose = np.array([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(pose[PoseSE3Index.SCALAR], [0.9])

    def test_vector_property(self):
        """Test that VECTOR slice indexes the qx, qy, qz components."""
        pose = np.array([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(pose[PoseSE3Index.VECTOR], [0.1, 0.2, 0.3])


class TestBoundingBoxSE3Index:
    """Tests for BoundingBoxSE3Index classproperties."""

    def test_scalar_property(self):
        """Test that SCALAR slice indexes the qw component."""
        bb = np.zeros(10)
        bb[BoundingBoxSE3Index.QW] = 1.0
        result = bb[BoundingBoxSE3Index.SCALAR]
        np.testing.assert_array_equal(result, [1.0])

    def test_vector_property(self):
        """Test that VECTOR slice indexes the qx, qy, qz components."""
        bb = np.zeros(10)
        bb[BoundingBoxSE3Index.QX] = 0.1
        bb[BoundingBoxSE3Index.QY] = 0.2
        bb[BoundingBoxSE3Index.QZ] = 0.3
        np.testing.assert_array_equal(bb[BoundingBoxSE3Index.VECTOR], [0.1, 0.2, 0.3])


class TestCorners3DIndex:
    """Tests for Corners3DIndex classproperties."""

    def test_bottom_slice(self):
        """Test that BOTTOM slice covers the four bottom corner indices."""
        corners = np.arange(8)
        bottom = corners[Corners3DIndex.BOTTOM]
        assert len(bottom) == 4
        np.testing.assert_array_equal(
            bottom,
            [
                Corners3DIndex.FRONT_LEFT_BOTTOM,
                Corners3DIndex.FRONT_RIGHT_BOTTOM,
                Corners3DIndex.BACK_RIGHT_BOTTOM,
                Corners3DIndex.BACK_LEFT_BOTTOM,
            ],
        )

    def test_top_slice(self):
        """Test that TOP slice covers the four top corner indices."""
        corners = np.arange(8)
        top = corners[Corners3DIndex.TOP]
        assert len(top) == 4
        np.testing.assert_array_equal(
            top,
            [
                Corners3DIndex.FRONT_LEFT_TOP,
                Corners3DIndex.FRONT_RIGHT_TOP,
                Corners3DIndex.BACK_RIGHT_TOP,
                Corners3DIndex.BACK_LEFT_TOP,
            ],
        )


class TestMatrixSO2Index:
    """Tests for MatrixSO2Index classproperties."""

    def test_x_axis(self):
        """Test that X_AXIS indexes the first column of a 2x2 matrix."""
        R = np.eye(2)
        np.testing.assert_array_equal(R[MatrixSO2Index.X_AXIS], [1.0, 0.0])

    def test_y_axis(self):
        """Test that Y_AXIS indexes the second column of a 2x2 matrix."""
        R = np.eye(2)
        np.testing.assert_array_equal(R[MatrixSO2Index.Y_AXIS], [0.0, 1.0])


class TestMatrixSO3Index:
    """Tests for MatrixSO3Index classproperties."""

    def test_x_axis(self):
        """Test that X_AXIS indexes the first column of a 3x3 matrix."""
        R = np.eye(3)
        np.testing.assert_array_equal(R[MatrixSO3Index.X_AXIS], [1.0, 0.0, 0.0])

    def test_y_axis(self):
        """Test that Y_AXIS indexes the second column of a 3x3 matrix."""
        R = np.eye(3)
        np.testing.assert_array_equal(R[MatrixSO3Index.Y_AXIS], [0.0, 1.0, 0.0])

    def test_z_axis(self):
        """Test that Z_AXIS indexes the third column of a 3x3 matrix."""
        R = np.eye(3)
        np.testing.assert_array_equal(R[MatrixSO3Index.Z_AXIS], [0.0, 0.0, 1.0])


class TestMatrixSE2Index:
    """Tests for MatrixSE2Index classproperties."""

    def test_rotation_block(self):
        """Test that ROTATION indexes the top-left 2x2 block."""
        M = np.eye(3)
        np.testing.assert_array_equal(M[MatrixSE2Index.ROTATION], np.eye(2))

    def test_translation_column(self):
        """Test that TRANSLATION indexes the top-right 2x1 column."""
        M = np.eye(3)
        M[0, 2] = 5.0
        M[1, 2] = 3.0
        np.testing.assert_array_equal(M[MatrixSE2Index.TRANSLATION], [5.0, 3.0])


class TestMatrixSE3Index:
    """Tests for MatrixSE3Index classproperties."""

    def test_rotation_block(self):
        """Test that ROTATION indexes the top-left 3x3 block."""
        M = np.eye(4)
        np.testing.assert_array_equal(M[MatrixSE3Index.ROTATION], np.eye(3))

    def test_translation_column(self):
        """Test that TRANSLATION indexes the top-right 3x1 column."""
        M = np.eye(4)
        M[0, 3] = 1.0
        M[1, 3] = 2.0
        M[2, 3] = 3.0
        np.testing.assert_array_equal(M[MatrixSE3Index.TRANSLATION], [1.0, 2.0, 3.0])
