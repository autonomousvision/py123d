import numpy as np

from py123d.conversion.sensor_io.lidar.laz_lidar_io import (
    encode_point_cloud_3d_as_laz_binary,
    is_laz_binary,
    load_point_cloud_3d_from_laz_binary,
)


class TestIsLazBinary:
    """Test LAZ binary format detection."""

    def test_valid_laz_binary(self):
        """Test that valid LAZ binary is detected correctly."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        assert is_laz_binary(laz_binary) is True

    def test_non_laz_binary(self):
        """Test that non-LAZ binary data is rejected."""
        assert is_laz_binary(b"not a laz file") is False

    def test_draco_binary_rejected(self):
        """Test that Draco binary is not detected as LAZ."""
        assert is_laz_binary(b"DRACO_DATA") is False

    def test_ipc_binary_rejected(self):
        """Test that Arrow IPC binary is not detected as LAZ."""
        assert is_laz_binary(b"ARROW1_DATA") is False


class TestLazRoundtrip:
    """Test LAZ encode/decode roundtrip."""

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves point cloud shape."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        assert decoded.shape == point_cloud.shape

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves float32 dtype."""
        point_cloud = np.random.rand(50, 3).astype(np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        assert decoded.dtype == np.float32

    def test_roundtrip_approximate_values(self):
        """Test that LAZ roundtrip preserves values approximately."""
        point_cloud = np.random.rand(100, 3).astype(np.float32) * 100.0
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        np.testing.assert_array_almost_equal(decoded, point_cloud, decimal=2)

    def test_roundtrip_single_point(self):
        """Test roundtrip with a single point."""
        point_cloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        assert decoded.shape == (1, 3)
        np.testing.assert_array_almost_equal(decoded, point_cloud, decimal=2)

    def test_roundtrip_large_point_cloud(self):
        """Test roundtrip with a large point cloud."""
        point_cloud = np.random.rand(10000, 3).astype(np.float32) * 50.0
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        assert decoded.shape == (10000, 3)

    def test_roundtrip_negative_coordinates(self):
        """Test roundtrip with negative coordinate values."""
        point_cloud = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 200.0
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        assert decoded.shape == point_cloud.shape
        np.testing.assert_array_almost_equal(decoded, point_cloud, decimal=2)

    def test_roundtrip_zeros(self):
        """Test roundtrip with all-zero point cloud."""
        point_cloud = np.zeros((10, 3), dtype=np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        decoded = load_point_cloud_3d_from_laz_binary(laz_binary)
        np.testing.assert_array_almost_equal(decoded, point_cloud, decimal=2)

    def test_encoded_is_compressed(self):
        """Test that LAZ encoding produces smaller data than raw numpy."""
        point_cloud = np.random.rand(1000, 3).astype(np.float32)
        laz_binary = encode_point_cloud_3d_as_laz_binary(point_cloud)
        # LAZ should compress the data (with some header overhead)
        assert isinstance(laz_binary, bytes)
        assert len(laz_binary) > 0
        assert len(laz_binary) < point_cloud.nbytes
