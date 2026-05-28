import numpy as np
import pytest

from py123d.geometry import PolylineSE3, PoseSE3
from py123d.geometry.utils.rotation_utils import (
    get_quaternion_array_from_euler_array,
    normalize_quaternion_array,
)


def _make_identity_poses(positions: np.ndarray) -> np.ndarray:
    """Helper: create SE3 array with identity rotations at given positions."""
    n = len(positions)
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, :3] = positions
    poses[:, 3] = 1.0  # qw = 1 (identity rotation)
    return poses


class TestPolylineSE3:
    """Tests for PolylineSE3."""

    def test_from_array(self):
        """Test creating PolylineSE3 from array."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        assert isinstance(polyline, PolylineSE3)
        assert polyline.array.shape == (2, 7)

    def test_from_array_copy(self):
        """Test that from_array copies the input by default."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        poses[0, 0] = 999.0
        assert polyline.array[0, 0] != 999.0

    def test_from_array_no_copy(self):
        """Test from_array with copy=False shares data."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses, copy=False)
        poses[0, 0] = 999.0
        assert polyline.array[0, 0] == 999.0

    def test_length(self):
        """Test translational path length."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        assert polyline.length == pytest.approx(2.0)

    def test_length_3d(self):
        """Test path length accounts for all 3 dimensions."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        assert polyline.length == pytest.approx(np.sqrt(3))

    def test_interpolate_scalar_returns_pose_se3(self):
        """Test scalar interpolation returns PoseSE3."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        result = polyline.interpolate(1.0)
        assert isinstance(result, PoseSE3)

    def test_interpolate_array_returns_ndarray(self):
        """Test array interpolation returns ndarray."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        result = polyline.interpolate(np.array([0.5, 1.0, 1.5]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 7)

    def test_interpolate_translation_midpoint(self):
        """Test that translation is correctly interpolated at midpoint."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 2, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        mid_dist = polyline.length / 2.0
        result = polyline.interpolate(mid_dist)
        assert isinstance(result, PoseSE3)
        assert result.x == pytest.approx(2.0, abs=1e-6)
        assert result.y == pytest.approx(1.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)

    def test_interpolate_identity_rotation_preserved(self):
        """Test that identity rotations remain identity throughout interpolation."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        result = polyline.interpolate(1.0)
        assert isinstance(result, PoseSE3)
        np.testing.assert_allclose(result.quaternion.array, [1, 0, 0, 0], atol=1e-10)

    def test_interpolate_rotation_midpoint(self):
        """Test rotation interpolation at midpoint of a 90-degree yaw rotation."""
        q_start = np.array([1.0, 0.0, 0.0, 0.0])
        q_end = get_quaternion_array_from_euler_array(np.array([0.0, 0.0, np.pi / 2]))
        poses = np.array(
            [
                [0, 0, 0, *q_start],
                [2, 0, 0, *q_end],
            ],
            dtype=np.float64,
        )
        polyline = PolylineSE3.from_array(poses)
        result = polyline.interpolate(1.0)
        assert isinstance(result, PoseSE3)
        # Midpoint should be ~45 degrees yaw
        assert result.yaw == pytest.approx(np.pi / 4, abs=1e-6)

    def test_interpolate_normalized(self):
        """Test normalized interpolation (0 to 1 range)."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        result = polyline.interpolate(0.5, normalized=True)
        assert isinstance(result, PoseSE3)
        assert result.x == pytest.approx(2.0, abs=1e-6)

    def test_interpolate_at_keyframes(self):
        """Test interpolation exactly at keyframe positions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = get_quaternion_array_from_euler_array(np.array([0.0, 0.0, np.pi / 2]))
        poses = np.array(
            [
                [0, 0, 0, *q1],
                [2, 0, 0, *q2],
            ],
            dtype=np.float64,
        )
        polyline = PolylineSE3.from_array(poses)
        # At distance ~0 (clamped to 1e-8)
        start = polyline.interpolate(0.0)
        assert isinstance(start, PoseSE3)
        assert start.x == pytest.approx(0.0, abs=1e-6)
        # At end
        end = polyline.interpolate(polyline.length)
        assert isinstance(end, PoseSE3)
        assert end.x == pytest.approx(2.0, abs=1e-6)
        assert end.yaw == pytest.approx(np.pi / 2, abs=1e-6)

    def test_interpolate_multi_segment(self):
        """Test interpolation across multiple segments."""
        poses = _make_identity_poses(
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=np.float64,
            )
        )
        polyline = PolylineSE3.from_array(poses)
        assert polyline.length == pytest.approx(3.0)
        # At distance 0.5 (within first segment)
        r1 = polyline.interpolate(0.5)
        assert isinstance(r1, PoseSE3)
        assert r1.x == pytest.approx(0.5, abs=1e-6)
        assert r1.y == pytest.approx(0.0, abs=1e-6)
        # At distance 1.5 (within second segment)
        r2 = polyline.interpolate(1.5)
        assert isinstance(r2, PoseSE3)
        assert r2.x == pytest.approx(1.0, abs=1e-6)
        assert r2.y == pytest.approx(0.5, abs=1e-6)
        # At distance 2.5 (within third segment)
        r3 = polyline.interpolate(2.5)
        assert isinstance(r3, PoseSE3)
        assert r3.z == pytest.approx(0.5, abs=1e-6)

    def test_nlerp_strategy(self):
        """Test that nlerp strategy works and produces valid results."""
        q_start = np.array([1.0, 0.0, 0.0, 0.0])
        q_end = get_quaternion_array_from_euler_array(np.array([0.0, 0.0, np.pi / 2]))
        poses = np.array(
            [
                [0, 0, 0, *q_start],
                [2, 0, 0, *q_end],
            ],
            dtype=np.float64,
        )
        polyline = PolylineSE3.from_array(poses, rotation_interpolation="nlerp")
        assert polyline.rotation_interpolation == "nlerp"
        result = polyline.interpolate(1.0)
        assert isinstance(result, PoseSE3)
        # NLERP midpoint should be close to SLERP midpoint for 90-degree rotation
        assert result.yaw == pytest.approx(np.pi / 4, abs=0.05)

    def test_slerp_vs_nlerp_small_angle(self):
        """SLERP and NLERP produce nearly identical results for small rotations."""
        q_start = np.array([1.0, 0.0, 0.0, 0.0])
        q_end = get_quaternion_array_from_euler_array(np.array([0.0, 0.0, 0.05]))
        poses = np.array(
            [
                [0, 0, 0, *q_start],
                [2, 0, 0, *q_end],
            ],
            dtype=np.float64,
        )
        slerp_polyline = PolylineSE3.from_array(poses, rotation_interpolation="slerp")
        nlerp_polyline = PolylineSE3.from_array(poses, rotation_interpolation="nlerp")
        slerp_result = slerp_polyline.interpolate(1.0)
        nlerp_result = nlerp_polyline.interpolate(1.0)
        assert isinstance(slerp_result, PoseSE3)
        assert isinstance(nlerp_result, PoseSE3)
        np.testing.assert_allclose(slerp_result.array, nlerp_result.array, atol=1e-4)

    def test_invalid_strategy_raises(self):
        """Test that an invalid rotation strategy raises an error."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        with pytest.raises(AssertionError, match="Unknown rotation interpolation"):
            PolylineSE3.from_array(poses, rotation_interpolation="invalid")

    def test_invalid_shape_raises(self):
        """Test that non-(N, 7) arrays raise an error."""
        with pytest.raises(AssertionError):
            PolylineSE3.from_array(np.zeros((3, 3)))

    def test_unit_quaternions_in_output(self):
        """Test that all interpolated quaternions are unit quaternions."""
        rng = np.random.default_rng(42)
        q1 = normalize_quaternion_array(rng.standard_normal(4))
        q2 = normalize_quaternion_array(rng.standard_normal(4))
        poses = np.array(
            [
                [0, 0, 0, *q1],
                [3, 4, 0, *q2],
            ],
            dtype=np.float64,
        )
        polyline = PolylineSE3.from_array(poses)
        distances = np.linspace(0.1, polyline.length, 20)
        result = polyline.interpolate(distances)
        assert isinstance(result, np.ndarray)
        quat_norms = np.linalg.norm(result[:, 3:], axis=-1)
        np.testing.assert_allclose(quat_norms, 1.0, atol=1e-10)

    def test_default_strategy_is_slerp(self):
        """Test that the default rotation interpolation is slerp."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        assert polyline.rotation_interpolation == "slerp"

    def test_subline_returns_polyline_se3(self):
        """subline returns a PolylineSE3 instance."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(1.0, 3.0)
        assert isinstance(sub, PolylineSE3)

    def test_subline_full_range(self):
        """subline(0, length) reproduces the original SE3 polyline (up to clip tolerance)."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(0.0, polyline.length)
        # Endpoints are interpolated at 1e-8 / length-1e-8 internally; tolerate that.
        np.testing.assert_allclose(sub.array, poses, atol=1e-7)
        assert sub.length == pytest.approx(polyline.length)

    def test_subline_partial_range(self):
        """subline cuts to a partial range with the expected length."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(1.0, 3.0)
        assert sub.length == pytest.approx(2.0)
        np.testing.assert_allclose(sub.array[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(sub.array[-1, :3], [3.0, 0.0, 0.0], atol=1e-6)

    def test_subline_exact_endpoints(self):
        """subline endpoints equal interpolate(start) / interpolate(end)."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 4]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        start_d, end_d = 1.0, polyline.length - 1.0
        sub = polyline.subline(start_d, end_d)
        np.testing.assert_allclose(sub.array[0], polyline.interpolate(start_d).array, atol=1e-9)
        np.testing.assert_allclose(sub.array[-1], polyline.interpolate(end_d).array, atol=1e-9)

    def test_subline_normalized(self):
        """Normalized distances produce the same result as absolute distances."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub_abs = polyline.subline(1.0, 3.0)
        sub_norm = polyline.subline(0.25, 0.75, normalized=True)
        np.testing.assert_allclose(sub_abs.array, sub_norm.array, atol=1e-9)

    def test_subline_clips_outside_range(self):
        """Distances outside [0, length] are clipped to the polyline bounds."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(-5.0, polyline.length + 5.0)
        np.testing.assert_allclose(sub.array, poses, atol=1e-7)

    def test_subline_swaps_reversed(self):
        """Reversed (start > end) is silently swapped."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        np.testing.assert_allclose(
            polyline.subline(3.0, 1.0).array,
            polyline.subline(1.0, 3.0).array,
            atol=1e-9,
        )

    def test_subline_raises_on_zero_length(self):
        """subline raises ValueError when start_distance == end_distance after clipping."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        with pytest.raises(ValueError, match="start_distance != end_distance"):
            polyline.subline(2.0, 2.0)

    def test_subline_preserves_intermediate_vertices(self):
        """Vertices strictly between start and end appear in the result."""
        poses = _make_identity_poses(np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(0.5, 3.5)
        assert sub.array.shape == (5, 7)
        np.testing.assert_allclose(sub.array[:, 0], [0.5, 1.0, 2.0, 3.0, 3.5], atol=1e-6)

    def test_subline_preserves_rotation_strategy(self):
        """The active rotation strategy propagates to the returned polyline."""
        poses = _make_identity_poses(np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64))
        polyline = PolylineSE3.from_array(poses, rotation_interpolation="nlerp")
        sub = polyline.subline(1.0, 3.0)
        assert sub.rotation_interpolation == "nlerp"

    def test_subline_quaternion_norms(self):
        """All quaternions in the resulting subline are unit norm."""
        rng = np.random.default_rng(42)
        q1 = normalize_quaternion_array(rng.standard_normal(4))
        q2 = normalize_quaternion_array(rng.standard_normal(4))
        poses = np.array([[0, 0, 0, *q1], [3, 4, 0, *q2]], dtype=np.float64)
        polyline = PolylineSE3.from_array(poses)
        sub = polyline.subline(polyline.length / 4, 3 * polyline.length / 4)
        quat_norms = np.linalg.norm(sub.array[:, 3:], axis=-1)
        np.testing.assert_allclose(quat_norms, 1.0, atol=1e-10)
