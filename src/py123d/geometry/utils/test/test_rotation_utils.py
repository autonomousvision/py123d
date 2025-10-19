import unittest

import numpy as np
import numpy.typing as npt


from py123d.geometry.utils.rotation_utils import (
    conjugate_quaternion_array,
    get_euler_array_from_quaternion_array,
)

# TODO @DanielDauner: Add more tests for the remaining functions
# from py123d.geometry.utils.rotation_utils import (
#     conjugate_quaternion_array,
#     get_euler_array_from_quaternion_array,
#     get_euler_array_from_rotation_matrices,
#     get_euler_array_from_rotation_matrix,
#     get_q_bar_matrices,
#     get_q_matrices,
#     get_quaternion_array_from_euler_array,
#     get_quaternion_array_from_rotation_matrices,
#     get_quaternion_array_from_rotation_matrix,
#     get_rotation_matrices_from_euler_array,
#     get_rotation_matrices_from_quaternion_array,
#     get_rotation_matrix_from_euler_array,
#     get_rotation_matrix_from_quaternion_array,
#     invert_quaternion_array,
#     multiply_quaternion_arrays,
#     normalize_angle,
#     normalize_quaternion_array,
# )


from pyquaternion import Quaternion as PyQuaternion


class TestRotationUtils(unittest.TestCase):

    def setUp(self):
        pass

    def _get_random_quaternion(self) -> npt.NDArray[np.float64]:
        random_quat: npt.NDArray[np.float64] = np.random.rand(4)
        random_quat /= np.linalg.norm(random_quat)
        return random_quat

    def _get_random_quaternion_array(self, n: int) -> npt.NDArray[np.float64]:
        random_quat_array: npt.NDArray[np.float64] = np.random.rand(n, 4)
        random_quat_array /= np.linalg.norm(random_quat_array, axis=1)[:, np.newaxis]
        return random_quat_array

    def test_conjugate_quaternion_array(self):
        for _ in range(10):
            random_quat = self._get_random_quaternion()
            conj_quat = conjugate_quaternion_array(random_quat)

            # Check if conjugation is correct
            np.testing.assert_allclose(
                conj_quat,
                np.array([random_quat[0], -random_quat[1], -random_quat[2], -random_quat[3]]),
                atol=1e-8,
            )

            # Check if double conjugation returns original quaternion
            double_conj_quat = conjugate_quaternion_array(conj_quat)
            np.testing.assert_allclose(
                double_conj_quat,
                random_quat,
                atol=1e-8,
            )

    def test_get_euler_array_from_quaternion_array(self):
        for _ in range(10):
            random_quat_array = self._get_random_quaternion_array(np.random.randint(0, 10))
            pyquaternions = [PyQuaternion(array=q) for q in random_quat_array]

            # Convert to Euler angles using our function
            euler_array = get_euler_array_from_quaternion_array(random_quat_array)

            # Test against pyquaternion results
            for i, pyq in enumerate(pyquaternions):
                # Convert to Euler angles using pyquaternion for comparison
                yaw, pitch, roll = pyq.yaw_pitch_roll
                euler_from_pyq = np.array([roll, pitch, yaw], dtype=np.float64)

                # Check if conversion is correct
                np.testing.assert_allclose(
                    euler_array[i],
                    euler_from_pyq,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
