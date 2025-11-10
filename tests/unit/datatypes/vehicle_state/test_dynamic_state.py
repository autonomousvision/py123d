import unittest

import numpy as np

from py123d.datatypes.vehicle_state.dynamic_state import (
    DynamicStateSE2,
    DynamicStateSE2Index,
    DynamicStateSE3,
    DynamicStateSE3Index,
)
from py123d.geometry import Vector2D, Vector3D


class TestDynamicStateSE3(unittest.TestCase):
    def test_init(self):
        velocity = Vector3D(1.0, 2.0, 3.0)
        acceleration = Vector3D(4.0, 5.0, 6.0)
        angular_velocity = Vector3D(7.0, 8.0, 9.0)

        state = DynamicStateSE3(velocity, acceleration, angular_velocity)

        assert np.allclose(state.velocity.array, velocity.array)
        assert np.allclose(state.acceleration.array, acceleration.array)
        assert np.allclose(state.angular_velocity.array, angular_velocity.array)

    def test_from_array(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        state = DynamicStateSE3.from_array(array)

        assert np.allclose(state.array, array)
        assert state.array is not array  # Default copy=True
        assert len(state.array) == len(DynamicStateSE3Index)

    def test_from_array_no_copy(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        state = DynamicStateSE3.from_array(array, copy=False)

        assert state.array is array

    def test_velocity_properties(self):
        velocity = Vector3D(1.0, 2.0, 3.0)
        state = DynamicStateSE3(velocity, Vector3D(0, 0, 0), Vector3D(0, 0, 0))

        assert np.allclose(state.velocity_3d.array, [1.0, 2.0, 3.0])
        assert np.allclose(state.velocity_2d.array, [1.0, 2.0])

    def test_acceleration_properties(self):
        acceleration = Vector3D(4.0, 5.0, 6.0)
        state = DynamicStateSE3(Vector3D(0, 0, 0), acceleration, Vector3D(0, 0, 0))

        assert np.allclose(state.acceleration_3d.array, [4.0, 5.0, 6.0])
        assert np.allclose(state.acceleration_2d.array, [4.0, 5.0])

    def test_dynamic_state_se2_projection(self):
        velocity = Vector3D(1.0, 2.0, 3.0)
        acceleration = Vector3D(4.0, 5.0, 6.0)
        angular_velocity = Vector3D(7.0, 8.0, 9.0)

        state_se3 = DynamicStateSE3(velocity, acceleration, angular_velocity)
        state_se2 = state_se3.dynamic_state_se2

        assert np.allclose(state_se2.velocity.array, [1.0, 2.0])
        assert np.allclose(state_se2.acceleration.array, [4.0, 5.0])
        assert np.isclose(state_se2.angular_velocity, 9.0)


class TestDynamicStateSE2(unittest.TestCase):
    def test_init(self):
        velocity = Vector2D(1.0, 2.0)
        acceleration = Vector2D(3.0, 4.0)
        angular_velocity = 5.0

        state = DynamicStateSE2(velocity, acceleration, angular_velocity)

        assert np.allclose(state.velocity.array, velocity.array)
        assert np.allclose(state.acceleration.array, acceleration.array)
        assert np.isclose(state.angular_velocity, angular_velocity)
        assert len(state.array) == len(DynamicStateSE2Index)

    def test_from_array(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = DynamicStateSE2.from_array(array)

        assert np.allclose(state.array, array)

    def test_velocity_properties(self):
        velocity = Vector2D(1.0, 2.0)
        state = DynamicStateSE2(velocity, Vector2D(0, 0), 0.0)

        assert np.allclose(state.velocity.array, [1.0, 2.0])
        assert np.allclose(state.velocity_2d.array, [1.0, 2.0])

    def test_acceleration_properties(self):
        acceleration = Vector2D(3.0, 4.0)
        state = DynamicStateSE2(Vector2D(0, 0), acceleration, 0.0)

        assert np.allclose(state.acceleration.array, [3.0, 4.0])
        assert np.allclose(state.acceleration_2d.array, [3.0, 4.0])

    def test_angular_velocity_property(self):
        state = DynamicStateSE2(Vector2D(0, 0), Vector2D(0, 0), 5.0)

        assert np.isclose(state.angular_velocity, 5.0)

    def test_array_property(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = DynamicStateSE2.from_array(array)

        assert np.array_equal(state.array, array)
