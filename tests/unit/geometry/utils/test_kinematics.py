import numpy as np

from py123d.geometry.utils.kinematics import (
    extract_linear_acceleration_from_se2,
    extract_linear_velocity_from_se2,
    extract_yaw_rate_from_se2,
    phase_unwrap,
)


class TestKinematicUtils:
    def test_phase_unwrap(self):
        """Tests the phase unwrapping function."""

        # Test case with jumps greater than pi
        yaws = np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, np.pi / 4, -np.pi / 2])
        expected_unwrapped = np.array([0.0, np.pi / 2, np.pi, 5 * np.pi / 4, 9 * np.pi / 4, 3 * np.pi / 2])
        unwrapped = phase_unwrap(yaws)
        assert np.allclose(unwrapped, expected_unwrapped), (
            "Phase unwrapping failed for yaws with jumps greater than pi."
        )

        # Test case without jumps greater than pi
        yaws = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        expected_unwrapped = yaws.copy()
        unwrapped = phase_unwrap(yaws)
        assert np.allclose(unwrapped, expected_unwrapped), (
            "Phase unwrapping altered yaws without jumps greater than pi."
        )

    def test_extract_linear_velocity_from_se2(self):
        """Tests the linear velocity extraction from SE2 poses."""

        # Test case with pure translation
        poses_se2 = np.array([[0, 0, 0], [0, 1, 0], [0, 4, 0]])
        expected_velocity = np.array([1.0, 3.0])
        velocity = extract_linear_velocity_from_se2(poses_se2, delta_t=1.0)
        assert np.allclose(velocity, expected_velocity), "Linear velocity extraction failed for pure translation."

    def test_extract_linear_acceleration_from_se2(self):
        """Tests the linear acceleration extraction from SE2 poses."""

        # Test case with constant acceleration
        poses_se2 = np.array([[0, 0, 0], [0, 1, 0], [0, 4, 0], [0, 9, 0]])
        expected_acceleration = np.array([2.0, 2.0])
        acceleration = extract_linear_acceleration_from_se2(poses_se2, delta_t=1.0)

        assert np.allclose(acceleration, expected_acceleration), (
            "Linear acceleration extraction failed for constant acceleration."
        )

    def test_extract_yaw_rate_from_se2(self):
        """Tests the yaw rate extraction from SE2 poses."""

        # Test case with constant yaw rate
        poses_se2 = np.array([[0, 0, 0], [0, 0, np.pi / 4], [0, 0, np.pi / 2], [0, 0, 3 * np.pi / 4]])
        expected_yaw_rate = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        yaw_rate = extract_yaw_rate_from_se2(poses_se2, delta_t=1.0)

        assert np.allclose(yaw_rate, expected_yaw_rate), "Yaw rate extraction failed for constant yaw rate."

        # Test case with phase wrapping
        poses_se2 = np.array([[0, 0, np.pi - 0.1], [0, 0, -np.pi + 0.1], [0, 0, -0.1], [0, 0, -np.pi + 0.1]])
        expected_yaw_rate = np.array([0.2, 0.2, 0.2])
        yaw_rate = extract_yaw_rate_from_se2(poses_se2, delta_t=1.0)
