import numpy as np

from py123d.conversion.sensor_io.camera.png_camera_io import (
    decode_image_from_png_binary,
    encode_image_as_png_binary,
    is_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)


class TestIsPngBinary:
    """Test PNG binary format detection."""

    def test_valid_png_binary(self):
        """Test that a valid PNG binary is detected correctly."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(image)
        assert is_png_binary(png_binary) is True

    def test_non_png_binary(self):
        """Test that non-PNG binary data is rejected."""
        assert is_png_binary(b"not a png") is False

    def test_jpeg_binary_rejected(self):
        """Test that JPEG binary is not detected as PNG."""
        jpeg_header = b"\xff\xd8some data\xff\xd9"
        assert is_png_binary(jpeg_header) is False

    def test_empty_bytes(self):
        """Test that empty bytes are rejected."""
        assert is_png_binary(b"") is False

    def test_partial_png_signature(self):
        """Test that partial PNG signature is rejected."""
        assert is_png_binary(b"\x89PNG") is False  # Missing remaining bytes


class TestEncodeDecodePng:
    """Test PNG encode/decode roundtrip.

    The contract: encode expects RGB input, decode returns RGB output.
    Since PNG is lossless, decode(encode(rgb_image)) should exactly equal the original.
    """

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.image_small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        self.image_medium = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_encode_returns_bytes(self):
        """Test that encoding returns bytes."""
        png_binary = encode_image_as_png_binary(self.image_small)
        assert isinstance(png_binary, bytes)

    def test_encode_returns_valid_png(self):
        """Test that encoded output is valid PNG binary."""
        png_binary = encode_image_as_png_binary(self.image_small)
        assert is_png_binary(png_binary)

    def test_roundtrip_preserves_shape(self):
        """Test that encode/decode roundtrip preserves image shape."""
        png_binary = encode_image_as_png_binary(self.image_medium)
        decoded = decode_image_from_png_binary(png_binary)
        assert decoded.shape == self.image_medium.shape

    def test_roundtrip_preserves_dtype(self):
        """Test that encode/decode roundtrip preserves uint8 dtype."""
        png_binary = encode_image_as_png_binary(self.image_small)
        decoded = decode_image_from_png_binary(png_binary)
        assert decoded.dtype == np.uint8

    def test_roundtrip_lossless(self):
        """Test that PNG roundtrip is exactly lossless.

        BUG 1: encode_image_as_png_binary does not convert RGB→BGR before cv2.imencode,
        but decode_image_from_png_binary converts BGR→RGB after cv2.imdecode.
        This means decode(encode(rgb_image)) has R and B channels swapped.
        """
        png_binary = encode_image_as_png_binary(self.image_small)
        decoded = decode_image_from_png_binary(png_binary)
        np.testing.assert_array_equal(decoded, self.image_small)

    def test_roundtrip_lossless_medium(self):
        """Test that PNG roundtrip is exactly lossless for medium images."""
        png_binary = encode_image_as_png_binary(self.image_medium)
        decoded = decode_image_from_png_binary(png_binary)
        np.testing.assert_array_equal(decoded, self.image_medium)

    def test_roundtrip_preserves_distinct_channels(self):
        """Test that R, G, B channels are preserved individually through roundtrip.

        BUG 1: Without RGB→BGR conversion in encode, the red and blue channels get swapped.
        """
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 200  # Red channel
        image[:, :, 1] = 100  # Green channel
        image[:, :, 2] = 50  # Blue channel

        png_binary = encode_image_as_png_binary(image)
        decoded = decode_image_from_png_binary(png_binary)

        np.testing.assert_array_equal(decoded[:, :, 0], 200, err_msg="Red channel must be preserved")
        np.testing.assert_array_equal(decoded[:, :, 1], 100, err_msg="Green channel must be preserved")
        np.testing.assert_array_equal(decoded[:, :, 2], 50, err_msg="Blue channel must be preserved")

    def test_encode_solid_color_image(self):
        """Test encoding a solid color image (R=G=B, so channel order doesn't matter)."""
        solid = np.full((64, 64, 3), 128, dtype=np.uint8)
        png_binary = encode_image_as_png_binary(solid)
        decoded = decode_image_from_png_binary(png_binary)
        np.testing.assert_array_equal(decoded, solid)

    def test_encode_black_image(self):
        """Test encoding a fully black image."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(black)
        decoded = decode_image_from_png_binary(png_binary)
        np.testing.assert_array_equal(decoded, black)

    def test_encode_white_image(self):
        """Test encoding a fully white image."""
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        png_binary = encode_image_as_png_binary(white)
        decoded = decode_image_from_png_binary(png_binary)
        np.testing.assert_array_equal(decoded, white)


class TestLoadPngFromFile:
    """Test loading PNG data from files."""

    def test_load_png_binary_from_file(self, tmp_path):
        """Test loading PNG binary from a file."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(image)
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_binary)

        loaded_binary = load_png_binary_from_png_file(png_path)
        assert loaded_binary == png_binary
        assert is_png_binary(loaded_binary)

    def test_load_image_from_file(self, tmp_path):
        """Test loading a numpy image from a PNG file."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(image)
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_binary)

        loaded_image = load_image_from_png_file(png_path)
        assert loaded_image.shape == image.shape
        assert loaded_image.dtype == np.uint8

    def test_encode_save_load_preserves_image(self, tmp_path):
        """Test the full pipeline: encode RGB, save to file, load back as RGB."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(image)
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_binary)

        loaded_image = load_image_from_png_file(png_path)
        # The full pipeline should preserve the original RGB image exactly
        np.testing.assert_array_equal(loaded_image, image)

    def test_load_binary_and_decode_match(self, tmp_path):
        """Test that loading binary then decoding matches loading image directly."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        png_binary = encode_image_as_png_binary(image)
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_binary)

        loaded_binary = load_png_binary_from_png_file(png_path)
        decoded_from_binary = decode_image_from_png_binary(loaded_binary)
        loaded_directly = load_image_from_png_file(png_path)

        np.testing.assert_array_equal(decoded_from_binary, loaded_directly)
