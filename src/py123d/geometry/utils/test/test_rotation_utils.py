from typing import Tuple
import unittest

import numpy as np
import numpy.typing as npt

from py123d.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex
from py123d.geometry.utils.rotation_utils import (
    conjugate_quaternion_array,
    get_euler_array_from_quaternion_array,
    get_euler_array_from_rotation_matrices,
    get_euler_array_from_rotation_matrix,
    get_q_bar_matrices,
    get_q_matrices,
    get_quaternion_array_from_euler_array,
    get_quaternion_array_from_rotation_matrices,
    get_quaternion_array_from_rotation_matrix,
    get_rotation_matrices_from_euler_array,
    get_rotation_matrices_from_quaternion_array,
    get_rotation_matrix_from_euler_array,
    get_rotation_matrix_from_quaternion_array,
    invert_quaternion_array,
    multiply_quaternion_arrays,
    normalize_angle,
    normalize_quaternion_array,
)


from pyquaternion import Quaternion as PyQuaternion


def _get_rotation_matrix_helper(euler_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Helper function to ensure ZYX (Yaw-Pitch-Roll) intrinsic Euler angle convention, aka Tait-Bryan angles.

    :param euler_array: Array of Euler angles [roll, pitch, yaw] in radians.
    :type euler_array: npt.NDArray[np.float64]
    :return: Rotation matrix corresponding to the given Euler angles.
    """

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(euler_array[EulerAnglesIndex.ROLL]), -np.sin(euler_array[EulerAnglesIndex.ROLL])],
            [0, np.sin(euler_array[EulerAnglesIndex.ROLL]), np.cos(euler_array[EulerAnglesIndex.ROLL])],
        ],
        dtype=np.float64,
    )
    R_y = np.array(
        [
            [np.cos(euler_array[EulerAnglesIndex.PITCH]), 0, np.sin(euler_array[EulerAnglesIndex.PITCH])],
            [0, 1, 0],
            [-np.sin(euler_array[EulerAnglesIndex.PITCH]), 0, np.cos(euler_array[EulerAnglesIndex.PITCH])],
        ],
        dtype=np.float64,
    )
    R_z = np.array(
        [
            [np.cos(euler_array[EulerAnglesIndex.YAW]), -np.sin(euler_array[EulerAnglesIndex.YAW]), 0],
            [np.sin(euler_array[EulerAnglesIndex.YAW]), np.cos(euler_array[EulerAnglesIndex.YAW]), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return R_z @ R_y @ R_x


class TestRotationUtils(unittest.TestCase):

    def setUp(self):
        pass

    def _get_random_quaternion(self) -> npt.NDArray[np.float64]:
        return PyQuaternion.random().q

    def _get_random_quaternion_array(self, n: int) -> npt.NDArray[np.float64]:
        random_quat_array = np.zeros((n, len(QuaternionIndex)), dtype=np.float64)
        for i in range(n):
            random_quat_array[i] = self._get_random_quaternion()
        return random_quat_array

    def _get_random_euler_array(self, n: int) -> npt.NDArray[np.float64]:
        random_euler_array: npt.NDArray[np.float64] = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            random_euler_array[i] = PyQuaternion.random().yaw_pitch_roll[
                ::-1
            ]  # Convert (yaw, pitch, roll) to (roll, pitch, yaw)
        return random_euler_array

    def test_conjugate_quaternion_array(self):
        """Test the conjugate_quaternion_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_flat = self._get_random_quaternion_array(N)

                random_quat = random_quat_flat.reshape(shape + (len(QuaternionIndex),))
                conj_quat = conjugate_quaternion_array(random_quat)

                np.testing.assert_allclose(
                    conj_quat[..., QuaternionIndex.QW],
                    random_quat[..., QuaternionIndex.QW],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    conj_quat[..., QuaternionIndex.QX],
                    -random_quat[..., QuaternionIndex.QX],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    conj_quat[..., QuaternionIndex.QY],
                    -random_quat[..., QuaternionIndex.QY],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    conj_quat[..., QuaternionIndex.QZ],
                    -random_quat[..., QuaternionIndex.QZ],
                    atol=1e-8,
                )

                # Check if double conjugation returns original quaternion
                double_conj_quat = conjugate_quaternion_array(conj_quat)
                np.testing.assert_allclose(
                    double_conj_quat,
                    random_quat,
                    atol=1e-8,
                )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 2, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            conjugate_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            conjugate_quaternion_array(invalid_quat)

    def test_get_euler_array_from_quaternion_array(self):
        """Test the get_euler_array_from_quaternion_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_array_flat = self._get_random_quaternion_array(N)
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Convert to Euler angles using our function
                euler_array = get_euler_array_from_quaternion_array(random_quat_array)

                euler_array_flat = euler_array.reshape((N, 3))
                # Test against pyquaternion results
                for i, q in enumerate(random_quat_array_flat):
                    pyq = PyQuaternion(array=q)
                    # Convert to Euler angles using pyquaternion for comparison
                    yaw, pitch, roll = pyq.yaw_pitch_roll
                    euler_from_pyq = np.array([roll, pitch, yaw], dtype=np.float64)

                    # Check if conversion is correct
                    np.testing.assert_allclose(euler_array_flat[i], euler_from_pyq, atol=1e-6)

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 2, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            get_euler_array_from_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            get_euler_array_from_quaternion_array(invalid_quat)

    def test_get_euler_array_from_rotation_matrices(self):
        """Test the get_euler_array_from_rotation_matrices function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                rotation_matrices_flat: npt.NDArray[np.float64] = np.zeros((N, 3, 3), dtype=np.float64)
                for i in range(N):
                    random_euler = self._get_random_euler_array(1)[0]
                    rotation_matrices_flat[i] = _get_rotation_matrix_helper(random_euler)

                rotation_matrices = rotation_matrices_flat.reshape(shape + (3, 3))

                # Convert to Euler angles using our function
                euler_array = get_euler_array_from_rotation_matrices(rotation_matrices)

                # Test against helper function results
                euler_array_flat = euler_array.reshape((N, 3))
                for i in range(N):
                    expected_rotation_matrix = _get_rotation_matrix_helper(euler_array_flat[i])
                    np.testing.assert_allclose(
                        rotation_matrices_flat[i],
                        expected_rotation_matrix,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 1))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((0, 3))  # (0, 3) rotation matrix shape (invalid)
            get_euler_array_from_rotation_matrices(invalid_rot)

        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3, 3, 8))  # (3, 3, 8) rotation matrix shape (invalid)
            get_euler_array_from_rotation_matrices(invalid_rot)

    def test_get_euler_array_from_rotation_matrix(self):
        """Test the get_euler_array_from_rotation_matrix function."""
        for _ in range(10):
            random_euler = self._get_random_euler_array(1)[0]
            rotation_matrix = _get_rotation_matrix_helper(random_euler)

            # Convert to Euler angles using our function
            euler_array = get_euler_array_from_rotation_matrix(rotation_matrix)

            # Check if conversion is correct
            np.testing.assert_allclose(
                euler_array,
                random_euler,
                atol=1e-8,
            )

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3,))  # (0, 3) rotation matrix shape (invalid)
            get_euler_array_from_rotation_matrix(invalid_rot)

        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3, 8))  # (3, 8) rotation matrix shape (invalid)
            get_euler_array_from_rotation_matrix(invalid_rot)

    def test_get_q_bar_matrices(self):
        """Test the get_q_bar_matrices function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_array_flat = self._get_random_quaternion_array(N)
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Compute Q_bar matrices using our function
                q_bar_matrices = get_q_bar_matrices(random_quat_array)

                q_bar_matrices_flat = q_bar_matrices.reshape((N, 4, 4))

                # Test against pyquaternion results
                for i, q in enumerate(random_quat_array_flat):
                    expected_q_bar = PyQuaternion(array=q)._q_bar_matrix()

                    # Check if Q_bar matrix is correct
                    np.testing.assert_allclose(
                        q_bar_matrices_flat[i],
                        expected_q_bar,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((3, 2))
        _test_by_shape((1, 2))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            get_q_bar_matrices(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            get_q_bar_matrices(invalid_quat)

    def test_get_q_matrices(self):

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_array_flat = self._get_random_quaternion_array(N)
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Compute Q matrices using our function
                q_matrices = get_q_matrices(random_quat_array)

                q_matrices_flat = q_matrices.reshape((N, 4, 4))

                # Test against pyquaternion results
                for i, q in enumerate(random_quat_array_flat):
                    expected_q = PyQuaternion(array=q)._q_matrix()

                    # Check if Q matrix is correct
                    np.testing.assert_allclose(
                        q_matrices_flat[i],
                        expected_q,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((3, 2))
        _test_by_shape((1, 2))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            get_q_matrices(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            get_q_matrices(invalid_quat)

    def test_get_quaternion_array_from_euler_array(self):
        """test the get_quaternion_array_from_euler_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_euler_array_flat = self._get_random_euler_array(N)
                random_euler_array = random_euler_array_flat.reshape(shape + (3,))

                # Convert to quaternion array using our function
                quat_array = get_quaternion_array_from_euler_array(random_euler_array)

                quat_array_flat = quat_array.reshape((N, len(QuaternionIndex)))

                # Test against pyquaternion results
                for i in range(N):
                    roll = random_euler_array_flat[i][EulerAnglesIndex.ROLL]
                    pitch = random_euler_array_flat[i][EulerAnglesIndex.PITCH]
                    yaw = random_euler_array_flat[i][EulerAnglesIndex.YAW]

                    pyquaternion = (
                        PyQuaternion(axis=[0, 0, 1], angle=yaw)
                        * PyQuaternion(axis=[0, 1, 0], angle=pitch)
                        * PyQuaternion(axis=[1, 0, 0], angle=roll)
                    )

                    expected_quat = pyquaternion.q

                    # Check if conversion is correct
                    np.testing.assert_allclose(
                        quat_array_flat[i],
                        expected_quat,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((3, 5))
        _test_by_shape((1, 0, 2))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((0,))  # Zero euler angles (invalid)
            get_quaternion_array_from_euler_array(invalid_euler)

        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((3, 8))  # Zero euler angles (invalid)
            get_quaternion_array_from_euler_array(invalid_euler)

    def test_get_quaternion_array_from_rotation_matrices(self):

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                rotation_matrices_flat: npt.NDArray[np.float64] = np.zeros((N, 3, 3), dtype=np.float64)
                for i in range(N):
                    random_euler = self._get_random_euler_array(1)[0]
                    rotation_matrices_flat[i] = _get_rotation_matrix_helper(random_euler)

                rotation_matrices = rotation_matrices_flat.reshape(shape + (3, 3))

                # Convert to quaternion array using our function
                quat_array = get_quaternion_array_from_rotation_matrices(rotation_matrices)

                quat_array_flat = quat_array.reshape((N, len(QuaternionIndex)))

                # Test against pyquaternion results
                for i in range(N):
                    expected_quaternion = PyQuaternion(matrix=rotation_matrices_flat[i]).q
                    actual_quaternion = quat_array_flat[i]

                    # Check if quaternions are equivalent (considering sign ambiguity)
                    # Quaternions q and -q represent the same rotation
                    np.testing.assert_equal(
                        np.allclose(actual_quaternion, expected_quaternion, atol=1e-8)
                        or np.allclose(actual_quaternion, -expected_quaternion, atol=1e-8),
                        True,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((3, 5))
        _test_by_shape((1, 0, 2))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((0, 3))  # (0, 3) rotation matrix shape (invalid)
            get_quaternion_array_from_rotation_matrices(invalid_rot)

        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3, 3, 8))  # (3, 3, 8) rotation matrix shape (invalid)
            get_quaternion_array_from_rotation_matrices(invalid_rot)

    def test_get_quaternion_array_from_rotation_matrix(self):
        """Test the get_quaternion_array_from_rotation_matrix function."""
        for _ in range(10):
            random_euler = self._get_random_euler_array(1)[0]
            rotation_matrix = _get_rotation_matrix_helper(random_euler)

            # Convert to quaternion array using our function
            quat_array = get_quaternion_array_from_rotation_matrix(rotation_matrix)

            expected_quaternion = PyQuaternion(matrix=rotation_matrix).q
            actual_quaternion = quat_array

            # Check if quaternions are equivalent (considering sign ambiguity)
            # Quaternions q and -q represent the same rotation
            np.testing.assert_equal(
                np.allclose(actual_quaternion, expected_quaternion, atol=1e-8)
                or np.allclose(actual_quaternion, -expected_quaternion, atol=1e-8),
                True,
            )

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3,))  # (0, 3) rotation matrix shape (invalid)
            get_quaternion_array_from_rotation_matrix(invalid_rot)

        with self.assertRaises(AssertionError):
            invalid_rot = np.zeros((3, 8))  # (3, 8) rotation matrix shape (invalid)
            get_quaternion_array_from_rotation_matrix(invalid_rot)

    def test_normalize_quaternion_array(self):
        """Test the normalize_quaternion_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                scale = np.random.uniform(0.1, 10.0)
                random_quat_array_flat = self._get_random_quaternion_array(N) * scale  # Scale to ensure non-unit norm
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Normalize using our function
                normalized_quat_array = normalize_quaternion_array(random_quat_array)

                normalized_quat_array_flat = normalized_quat_array.reshape((N, len(QuaternionIndex)))

                # Check if each quaternion is normalized
                for i in range(N):
                    norm = np.linalg.norm(normalized_quat_array_flat[i])
                    np.testing.assert_allclose(norm, 1.0, atol=1e-8)

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 5, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            normalize_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            normalize_quaternion_array(invalid_quat)

    def test_get_rotation_matrices_from_euler_array(self):
        """Test the get_rotation_matrices_from_euler_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_euler_array_flat = self._get_random_euler_array(N)
                random_euler_array = random_euler_array_flat.reshape(shape + (3,))

                # Convert to rotation matrices using our function
                rotation_matrices = get_rotation_matrices_from_euler_array(random_euler_array)

                rotation_matrices_flat = rotation_matrices.reshape((N, 3, 3))

                # Test against helper function results
                for i in range(N):
                    expected_rotation_matrix = _get_rotation_matrix_helper(random_euler_array_flat[i])
                    np.testing.assert_allclose(
                        rotation_matrices_flat[i],
                        expected_rotation_matrix,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 1))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((0, 5))  # Zero euler angles (invalid)
            get_rotation_matrices_from_euler_array(invalid_euler)

        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((3, 8))  # Zero euler angles (invalid)
            get_rotation_matrices_from_euler_array(invalid_euler)

    def test_get_rotation_matrices_from_quaternion_array(self):
        """Test the get_rotation_matrices_from_quaternion_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_array_flat = self._get_random_quaternion_array(N)
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Convert to rotation matrices using our function
                rotation_matrices = get_rotation_matrices_from_quaternion_array(random_quat_array)

                rotation_matrices_flat = rotation_matrices.reshape((N, 3, 3))

                # Test against pyquaternion results
                for i, q in enumerate(random_quat_array_flat):
                    expected_rotation_matrix = PyQuaternion(array=q).rotation_matrix

                    # Check if rotation matrix is correct
                    np.testing.assert_allclose(
                        rotation_matrices_flat[i],
                        expected_rotation_matrix,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 5, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            get_rotation_matrices_from_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            get_rotation_matrices_from_quaternion_array(invalid_quat)

    def test_get_rotation_matrix_from_euler_array(self):
        """Test the get_rotation_matrix_from_euler_array function."""
        for _ in range(10):
            random_euler = self._get_random_euler_array(1)[0]

            # Convert to rotation matrix using our function
            rotation_matrix = get_rotation_matrix_from_euler_array(random_euler)

            expected_rotation_matrix = _get_rotation_matrix_helper(random_euler)

            # Check if conversion is correct
            np.testing.assert_allclose(
                rotation_matrix,
                expected_rotation_matrix,
                atol=1e-8,
            )

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((0,))  # Zero euler angles (invalid)
            get_rotation_matrix_from_euler_array(invalid_euler)

        with self.assertRaises(AssertionError):
            invalid_euler = np.zeros((8,))  # Zero euler angles (invalid)
            get_rotation_matrix_from_euler_array(invalid_euler)

    def test_get_rotation_matrix_from_quaternion_array(self):
        """Test the get_rotation_matrix_from_quaternion_array function."""
        for _ in range(10):
            random_quat = self._get_random_quaternion()

            # Convert to rotation matrix using our function
            rotation_matrix = get_rotation_matrix_from_quaternion_array(random_quat)

            expected_rotation_matrix = PyQuaternion(array=random_quat).rotation_matrix

            # Check if conversion is correct
            np.testing.assert_allclose(
                rotation_matrix,
                expected_rotation_matrix,
                atol=1e-8,
            )

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            get_rotation_matrix_from_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((8,))  # Zero quaternion (invalid)
            get_rotation_matrix_from_quaternion_array(invalid_quat)

    def test_invert_quaternion_array(self):
        """Test the invert_quaternion_array function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_quat_array_flat = self._get_random_quaternion_array(N)
                random_quat_array = random_quat_array_flat.reshape(shape + (len(QuaternionIndex),))

                # Invert using our function
                inverted_quat_array = invert_quaternion_array(random_quat_array)

                inverted_quat_array_flat = inverted_quat_array.reshape((N, len(QuaternionIndex)))

                # Test against pyquaternion results
                for i, q in enumerate(random_quat_array_flat):
                    pyq = PyQuaternion(array=q)
                    expected_inverse = pyq.inverse.q

                    # Check if inversion is correct
                    np.testing.assert_allclose(
                        inverted_quat_array_flat[i],
                        expected_inverse,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 5, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((0,))  # Zero quaternion (invalid)
            invert_quaternion_array(invalid_quat)

        with self.assertRaises(AssertionError):
            invalid_quat = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            invert_quaternion_array(invalid_quat)

    def test_multiply_quaternion_arrays(self):
        """Test the multiply_quaternion_arrays function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                quat_array1_flat = self._get_random_quaternion_array(N)
                quat_array2_flat = self._get_random_quaternion_array(N)

                quat_array1 = quat_array1_flat.reshape(shape + (len(QuaternionIndex),))
                quat_array2 = quat_array2_flat.reshape(shape + (len(QuaternionIndex),))

                # Multiply using our function
                multiplied_quat_array = multiply_quaternion_arrays(quat_array1, quat_array2)

                multiplied_quat_array_flat = multiplied_quat_array.reshape((N, len(QuaternionIndex)))

                # Test against pyquaternion results
                for i in range(N):
                    pyq1 = PyQuaternion(array=quat_array1_flat[i])
                    pyq2 = PyQuaternion(array=quat_array2_flat[i])
                    expected_product = (pyq1 * pyq2).q

                    # Check if multiplication is correct
                    np.testing.assert_allclose(
                        multiplied_quat_array_flat[i],
                        expected_product,
                        atol=1e-8,
                    )

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 5, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test invalid input
        with self.assertRaises(AssertionError):
            invalid_quat1 = np.zeros((0,))  # Zero quaternion (invalid)
            invalid_quat2 = np.zeros((0,))  # Zero quaternion (invalid)
            multiply_quaternion_arrays(invalid_quat1, invalid_quat2)

        with self.assertRaises(AssertionError):
            invalid_quat1 = np.zeros((len(QuaternionIndex), 8))  # Zero quaternion (invalid)
            invalid_quat2 = np.zeros((len(QuaternionIndex), 4))  # Zero quaternion (invalid)
            multiply_quaternion_arrays(invalid_quat1, invalid_quat2)

    def test_normalize_angle(self):
        """Test the normalize_angle function."""

        def _test_by_shape(shape: Tuple[int, ...]) -> None:
            for _ in range(10):
                N = np.prod(shape)
                random_angles_flat = np.random.uniform(-10 * np.pi, 10 * np.pi, size=N)
                random_angles = random_angles_flat.reshape(shape)

                # Normalize using our function
                normalized_angles = normalize_angle(random_angles)

                normalized_angles_flat = normalized_angles.reshape((N,))

                # Check if each angle is within [-pi, pi]
                for i in range(N):
                    angle = normalized_angles_flat[i]
                    self.assertGreaterEqual(angle, -np.pi - 1e-8)
                    self.assertLessEqual(angle, np.pi + 1e-8)

        # Test single-dim shape
        _test_by_shape((1,))

        # Test multi-dim shape
        _test_by_shape((2, 3))
        _test_by_shape((1, 5, 3))

        # Test zero-dim shape
        _test_by_shape((0,))

        # Test float
        with self.subTest("Test float input"):
            angle = 4 * np.pi
            normalized_angle = normalize_angle(angle)
            self.assertGreaterEqual(normalized_angle, -np.pi - 1e-8)
            self.assertLessEqual(normalized_angle, np.pi + 1e-8)


if __name__ == "__main__":
    unittest.main()
