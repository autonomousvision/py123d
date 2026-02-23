import numpy as np
import pytest

from py123d.conversion.sensor_io.lidar.ipc_lidar_io import (
    encode_point_cloud_3d_as_ipc_binary,
    encode_point_cloud_features_as_ipc_binary,
    is_ipc_binary,
    load_point_cloud_3d_from_ipc_binary,
    load_point_cloud_features_from_ipc_binary,
)


class TestIsIpcBinary:
    """Test Arrow IPC binary format detection."""

    def test_detects_encoded_point_cloud(self):
        """Test that is_ipc_binary detects output of encode_point_cloud_3d_as_ipc_binary."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        assert is_ipc_binary(ipc_binary) is True

    def test_detects_encoded_features(self):
        """Test that is_ipc_binary detects output of encode_point_cloud_features_as_ipc_binary."""
        features = {"intensity": np.random.rand(100).astype(np.float32)}
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features)
        assert is_ipc_binary(ipc_binary) is True

    def test_detects_zstd_encoded(self):
        """Test that is_ipc_binary detects zstd-compressed IPC output."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec="zstd")
        assert is_ipc_binary(ipc_binary) is True

    def test_detects_lz4_encoded(self):
        """Test that is_ipc_binary detects lz4-compressed IPC output."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec="lz4")
        assert is_ipc_binary(ipc_binary) is True

    def test_detects_uncompressed_encoded(self):
        """Test that is_ipc_binary detects uncompressed IPC output."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec=None)
        assert is_ipc_binary(ipc_binary) is True

    def test_non_ipc_binary(self):
        """Test that non-IPC binary data is rejected."""
        assert is_ipc_binary(b"not an ipc file") is False

    def test_draco_binary_rejected(self):
        """Test that Draco binary is not detected as IPC."""
        assert is_ipc_binary(b"DRACO_DATA") is False

    def test_laz_binary_rejected(self):
        """Test that LAZ binary is not detected as IPC."""
        assert is_ipc_binary(b"LASF_DATA") is False

    def test_empty_bytes(self):
        """Test that empty bytes are rejected."""
        assert is_ipc_binary(b"") is False


class TestIpcPointCloud3dRoundtrip:
    """Test Arrow IPC point cloud 3D encode/decode roundtrip."""

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves point cloud shape."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        assert decoded.shape == point_cloud.shape

    def test_roundtrip_exact_values(self):
        """Test that IPC roundtrip preserves values exactly (lossless)."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_with_zstd_compression(self):
        """Test roundtrip with zstd compression."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec="zstd")
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_with_lz4_compression(self):
        """Test roundtrip with lz4 compression."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec="lz4")
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_without_compression(self):
        """Test roundtrip without compression."""
        point_cloud = np.random.rand(100, 3).astype(np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud, codec=None)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_single_point(self):
        """Test roundtrip with a single point."""
        point_cloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_large_point_cloud(self):
        """Test roundtrip with a large point cloud."""
        point_cloud = np.random.rand(10000, 3).astype(np.float32) * 50.0
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        assert decoded.shape == (10000, 3)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_negative_coordinates(self):
        """Test roundtrip with negative coordinate values."""
        point_cloud = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 200.0
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_roundtrip_zeros(self):
        """Test roundtrip with all-zero point cloud."""
        point_cloud = np.zeros((10, 3), dtype=np.float32)
        ipc_binary = encode_point_cloud_3d_as_ipc_binary(point_cloud)
        decoded = load_point_cloud_3d_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded, point_cloud)

    def test_invalid_shape_2d_wrong_columns(self):
        """Test that a 2D array with wrong number of columns raises assertion."""
        point_cloud = np.random.rand(100, 4).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_3d_as_ipc_binary(point_cloud)

    def test_invalid_shape_1d(self):
        """Test that a 1D array raises assertion."""
        point_cloud = np.random.rand(100).astype(np.float32)
        with pytest.raises(AssertionError):
            encode_point_cloud_3d_as_ipc_binary(point_cloud)


class TestIpcPointCloudFeaturesRoundtrip:
    """Test Arrow IPC point cloud features encode/decode roundtrip."""

    def test_roundtrip_single_feature(self):
        """Test roundtrip with a single feature."""
        features = {"intensity": np.random.rand(100).astype(np.float32)}
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features)
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)

        assert "intensity" in decoded
        np.testing.assert_array_equal(decoded["intensity"], features["intensity"])

    def test_roundtrip_multiple_features(self):
        """Test roundtrip with multiple features."""
        features = {
            "intensity": np.random.rand(100).astype(np.float32),
            "ring": np.random.randint(0, 64, 100).astype(np.int32),
            "timestamp": np.random.rand(100).astype(np.float64),
        }
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features)
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)

        assert set(decoded.keys()) == set(features.keys())
        for key in features.keys():  # noqa: PLC0206
            np.testing.assert_array_equal(decoded[key], features[key])

    def test_roundtrip_with_zstd(self):
        """Test feature roundtrip with zstd compression."""
        features = {"intensity": np.random.rand(100).astype(np.float32)}
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features, codec="zstd")
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded["intensity"], features["intensity"])

    def test_roundtrip_with_lz4(self):
        """Test feature roundtrip with lz4 compression."""
        features = {"intensity": np.random.rand(100).astype(np.float32)}
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features, codec="lz4")
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded["intensity"], features["intensity"])

    def test_roundtrip_without_compression(self):
        """Test feature roundtrip without compression."""
        features = {"intensity": np.random.rand(100).astype(np.float32)}
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features, codec=None)
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)
        np.testing.assert_array_equal(decoded["intensity"], features["intensity"])

    def test_roundtrip_preserves_feature_names(self):
        """Test that feature names are preserved through roundtrip."""
        features = {
            "intensity": np.random.rand(50).astype(np.float32),
            "elongation": np.random.rand(50).astype(np.float32),
            "range": np.random.rand(50).astype(np.float32),
            "lidar_id": np.zeros(50, dtype=np.int32),
        }
        ipc_binary = encode_point_cloud_features_as_ipc_binary(features)
        decoded = load_point_cloud_features_from_ipc_binary(ipc_binary)
        assert set(decoded.keys()) == set(features.keys())
