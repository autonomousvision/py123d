import numpy as np
import pytest

from py123d.conversion.sensor_io.lidar.draco_lidar_io import (
    DRACO_QUANTIZATION_BITS,
    encode_point_cloud_3d_as_draco_binary,
    is_draco_binary,
    load_point_cloud_3d_from_draco_binary,
)


class TestIsDracoBinary:
    """Test Draco binary format detection."""

    def test_valid_draco_binary(self):
        """Test that valid Draco binary is detected correctly."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        assert is_draco_binary(draco_binary) is True

    def test_non_draco_binary(self):
        """Test that non-Draco binary data is rejected."""
        assert is_draco_binary(b"not a draco file") is False

    def test_laz_binary_rejected(self):
        """Test that LAZ binary is not detected as Draco."""
        assert is_draco_binary(b"LASF_DATA") is False

    def test_ipc_binary_rejected(self):
        """Test that Arrow IPC binary is not detected as Draco."""
        assert is_draco_binary(b"ARROW1_DATA") is False

    def test_empty_bytes(self):
        """Test that empty bytes are rejected."""
        assert is_draco_binary(b"") is False


class TestDracoRoundtrip:
    """Test Draco encode/decode roundtrip."""

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves point cloud shape."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        assert decoded.shape == point_cloud.shape

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves float32 dtype."""
        point_cloud = np.random.rand(50, 3).astype(np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        assert decoded.dtype == np.float32

    def test_roundtrip_approximate_values(self):
        """Test that Draco roundtrip preserves point values within quantization tolerance."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        # DRACO_PRESERVE_ORDER=False means points may be reordered.
        # Sort both arrays by coordinates before comparing.
        original_sorted = point_cloud[np.lexsort(point_cloud.T)]
        decoded_sorted = decoded[np.lexsort(decoded.T)]
        # 14-bit quantization over the value range introduces small errors
        np.testing.assert_array_almost_equal(decoded_sorted, original_sorted, decimal=3)

    def test_roundtrip_single_point(self):
        """Test roundtrip with a single point."""
        point_cloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        assert decoded.shape == (1, 3)

    def test_roundtrip_large_point_cloud(self):
        """Test roundtrip with a large point cloud."""
        point_cloud = np.random.rand(10000, 3).astype(np.float32) * 50.0
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        assert decoded.shape == (10000, 3)

    def test_roundtrip_negative_coordinates(self):
        """Test roundtrip with negative coordinate values."""
        point_cloud = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 200.0
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        decoded = load_point_cloud_3d_from_draco_binary(draco_binary)
        assert decoded.shape == point_cloud.shape

    def test_encoded_is_compressed(self):
        """Test that Draco encoding produces smaller data than raw numpy."""
        point_cloud = np.random.rand(1000, 3).astype(np.float32)
        draco_binary = encode_point_cloud_3d_as_draco_binary(point_cloud)
        raw_size = point_cloud.nbytes
        # Draco should compress significantly
        assert len(draco_binary) < raw_size

    def test_invalid_shape_2d_wrong_columns(self):
        """Test that a 2D array with wrong number of columns raises assertion."""
        point_cloud = np.random.rand(100, 4).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_3d_as_draco_binary(point_cloud)

    def test_invalid_shape_1d(self):
        """Test that a 1D array raises assertion."""
        point_cloud = np.random.rand(100).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_3d_as_draco_binary(point_cloud)

    def test_quantization_bits_constant(self):
        """Test that quantization bits constant has expected value."""
        assert DRACO_QUANTIZATION_BITS == 14
