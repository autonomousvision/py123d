import numpy as np

from py123d.conversion.sensor_io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    is_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)


class TestIsJpegBinary:
    """Test JPEG binary format detection."""

    def test_valid_jpeg_binary(self):
        """Test that a valid JPEG binary is detected correctly."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(image)
        assert is_jpeg_binary(jpeg_binary) is True

    def test_non_jpeg_binary(self):
        """Test that non-JPEG binary data is rejected."""
        assert is_jpeg_binary(b"not a jpeg") is False

    def test_png_binary_rejected(self):
        """Test that PNG binary is not detected as JPEG."""
        png_header = b"\x89PNG\r\n\x1a\nsome data"
        assert is_jpeg_binary(png_header) is False

    def test_empty_bytes(self):
        """Test that empty bytes are rejected."""
        assert is_jpeg_binary(b"") is False

    def test_jpeg_soi_marker_only(self):
        """Test that having only the SOI marker without EOI is rejected."""
        assert is_jpeg_binary(b"\xff\xd8some data") is False

    def test_jpeg_eoi_marker_only(self):
        """Test that having only the EOI marker without SOI is rejected."""
        assert is_jpeg_binary(b"some data\xff\xd9") is False


class TestEncodeDecodeJpeg:
    """Test JPEG encode/decode roundtrip.

    The contract: encode expects RGB input, decode returns RGB output.
    Roundtrip decode(encode(rgb_image)) should approximately preserve the original RGB image.
    """

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.image_small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        self.image_medium = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.image_large = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def test_encode_returns_bytes(self):
        """Test that encoding returns bytes."""
        jpeg_binary = encode_image_as_jpeg_binary(self.image_small)
        assert isinstance(jpeg_binary, bytes)

    def test_encode_returns_valid_jpeg(self):
        """Test that encoded output is valid JPEG binary."""
        jpeg_binary = encode_image_as_jpeg_binary(self.image_small)
        assert is_jpeg_binary(jpeg_binary)

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves image shape."""
        jpeg_binary = encode_image_as_jpeg_binary(self.image_medium)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        assert decoded.shape == self.image_medium.shape

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves uint8 dtype."""
        jpeg_binary = encode_image_as_jpeg_binary(self.image_small)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        assert decoded.dtype == np.uint8

    def test_roundtrip_preserves_colors(self):
        """Test that JPEG roundtrip preserves RGB pixel values approximately.

        BUG 1: encode_image_as_jpeg_binary does not convert RGB→BGR before cv2.imencode,
        but decode_image_from_jpeg_binary converts BGR→RGB after cv2.imdecode.
        This means decode(encode(rgb_image)) swaps R and B channels.
        """
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel only
        image[:, :, 1] = 0
        image[:, :, 2] = 0

        jpeg_binary = encode_image_as_jpeg_binary(image)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)

        # After roundtrip, the red channel should still be dominant
        assert decoded[:, :, 0].mean() > 200, "Red channel should be preserved after roundtrip"
        assert decoded[:, :, 2].mean() < 50, "Blue channel should remain near zero after roundtrip"

    def test_roundtrip_approximate_values(self):
        """Test that JPEG roundtrip preserves values approximately (lossy compression)."""
        # Use a smooth gradient image (representative of real camera data) rather than
        # random noise, which is the worst case for JPEG's DCT-based compression.
        gradient = np.zeros((64, 64, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        gradient[:, :, 1] = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]
        gradient[:, :, 2] = 128

        jpeg_binary = encode_image_as_jpeg_binary(gradient)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        # JPEG is lossy, but mean difference should be small for smooth images
        assert np.mean(np.abs(decoded.astype(float) - gradient.astype(float))) < 10.0

    def test_encode_various_sizes(self):
        """Test encoding images of various sizes."""
        for image in [self.image_small, self.image_medium, self.image_large]:
            jpeg_binary = encode_image_as_jpeg_binary(image)
            assert is_jpeg_binary(jpeg_binary)
            decoded = decode_image_from_jpeg_binary(jpeg_binary)
            assert decoded.shape == image.shape

    def test_encode_solid_color_image(self):
        """Test encoding a solid color image."""
        solid = np.full((64, 64, 3), 128, dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(solid)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        assert decoded.shape == solid.shape

    def test_encode_black_image(self):
        """Test encoding a fully black image."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(black)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        assert decoded.shape == black.shape
        np.testing.assert_array_almost_equal(decoded, black, decimal=0)

    def test_encode_white_image(self):
        """Test encoding a fully white image."""
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(white)
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        assert decoded.shape == white.shape


class TestLoadJpegFromFile:
    """Test loading JPEG data from files."""

    def test_load_jpeg_binary_from_file(self, tmp_path):
        """Test loading JPEG binary from a file."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(image)
        jpeg_path = tmp_path / "test.jpg"
        jpeg_path.write_bytes(jpeg_binary)

        loaded_binary = load_jpeg_binary_from_jpeg_file(jpeg_path)
        assert loaded_binary == jpeg_binary
        assert is_jpeg_binary(loaded_binary)

    def test_load_image_from_file(self, tmp_path):
        """Test loading a numpy image from a JPEG file."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(image)
        jpeg_path = tmp_path / "test.jpg"
        jpeg_path.write_bytes(jpeg_binary)

        loaded_image = load_image_from_jpeg_file(jpeg_path)
        assert loaded_image.shape == image.shape
        assert loaded_image.dtype == np.uint8

    def test_load_binary_and_decode_match(self, tmp_path):
        """Test that loading binary then decoding matches loading image directly."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        jpeg_binary = encode_image_as_jpeg_binary(image)
        jpeg_path = tmp_path / "test.jpg"
        jpeg_path.write_bytes(jpeg_binary)

        loaded_binary = load_jpeg_binary_from_jpeg_file(jpeg_path)
        decoded_from_binary = decode_image_from_jpeg_binary(loaded_binary)
        loaded_directly = load_image_from_jpeg_file(jpeg_path)

        np.testing.assert_array_equal(decoded_from_binary, loaded_directly)

    def test_encode_then_load_preserves_colors(self, tmp_path):
        """Test that encoding an RGB image, saving to file, and loading back preserves colors.

        BUG 1: encode does not convert RGB→BGR, so the saved JPEG has swapped channels.
        load_image_from_jpeg_file reads with cv2.imread (BGR) then converts BGR→RGB,
        which double-inverts relative to the broken encode, masking the bug for this path.
        But decode_image_from_jpeg_binary only does one BGR→RGB conversion, exposing it.
        """
        # Create image with distinct R, G, B channels
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 200  # Red
        image[:, :, 1] = 50  # Green
        image[:, :, 2] = 10  # Blue

        jpeg_binary = encode_image_as_jpeg_binary(image)
        jpeg_path = tmp_path / "test.jpg"
        jpeg_path.write_bytes(jpeg_binary)

        # decode_image_from_jpeg_binary should return the same as load_image_from_jpeg_file
        decoded = decode_image_from_jpeg_binary(jpeg_binary)
        loaded = load_image_from_jpeg_file(jpeg_path)
        np.testing.assert_array_equal(decoded, loaded)

        # Both should approximately match the original RGB image
        assert decoded[:, :, 0].mean() > 150, "Red channel should be dominant after decode"
        assert decoded[:, :, 2].mean() < 60, "Blue channel should stay low after decode"
