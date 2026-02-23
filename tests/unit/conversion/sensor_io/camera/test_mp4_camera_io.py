import numpy as np
import pytest

from py123d.conversion.sensor_io.camera.mp4_camera_io import MP4Reader, MP4Writer, get_mp4_reader_from_path


class TestMP4Writer:
    """Test MP4Writer functionality."""

    def test_write_single_frame(self, tmp_path):
        """Test writing a single frame to an MP4 file."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame_idx = writer.write_frame(frame)
        writer.close()

        assert frame_idx == 0
        assert output_path.exists()

    def test_write_multiple_frames(self, tmp_path):
        """Test writing multiple frames and tracking frame count."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)

        for i in range(10):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            frame_idx = writer.write_frame(frame)
            assert frame_idx == i

        assert writer.frame_count == 10
        writer.close()

    def test_write_frame_size_mismatch(self, tmp_path):
        """Test that writing a frame with different size raises ValueError."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)

        frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write_frame(frame1)

        frame2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="doesn't match video size"):
            writer.write_frame(frame2)

        writer.close()

    def test_close_releases_writer(self, tmp_path):
        """Test that close releases the video writer."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()

        assert writer.writer is None

    def test_close_without_writing(self, tmp_path):
        """Test that closing without writing any frames does not error."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)
        writer.close()
        assert writer.writer is None

    def test_creates_parent_directories(self, tmp_path):
        """Test that nested output directories are created automatically."""
        output_path = tmp_path / "nested" / "dir" / "test.mp4"
        writer = MP4Writer(output_path, fps=30.0)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()

        assert output_path.exists()


class TestMP4Reader:
    """Test MP4Reader functionality."""

    @pytest.fixture
    def mp4_file(self, tmp_path):
        """Create a test MP4 file with known frames."""
        output_path = tmp_path / "test.mp4"
        writer = MP4Writer(output_path, fps=10.0)
        frames = []
        for i in range(5):
            frame = np.full((64, 64, 3), i * 50, dtype=np.uint8)
            frames.append(frame)
            writer.write_frame(frame)
        writer.close()
        return output_path, frames

    def test_reader_properties(self, mp4_file):
        """Test that reader exposes correct video properties."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path))

        assert reader.frame_count == 5
        assert reader.fps == pytest.approx(10.0, abs=1.0)
        assert reader.width == 64
        assert reader.height == 64

    def test_get_frame_returns_numpy_array(self, mp4_file):
        """Test that get_frame returns a numpy array."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path))

        frame = reader.get_frame(0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_get_frame_out_of_range(self, mp4_file):
        """Test that out-of-range frame index raises IndexError (not AttributeError).

        BUG 3: MP4Reader.get_frame line 104 references self.frames which is undefined
        when read_all=False. This causes AttributeError instead of IndexError.
        """
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path))  # read_all=False by default

        with pytest.raises(IndexError):
            reader.get_frame(100)

    def test_get_frame_negative_index(self, mp4_file):
        """Test that negative frame index raises IndexError (not AttributeError).

        BUG 3: Same self.frames reference issue as test_get_frame_out_of_range.
        """
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path))  # read_all=False by default

        with pytest.raises(IndexError):
            reader.get_frame(-1)

    def test_get_frame_out_of_range_read_all(self, mp4_file):
        """Test that out-of-range frame index raises IndexError in read_all mode."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path), read_all=True)

        with pytest.raises(IndexError):
            reader.get_frame(100)

    def test_get_frame_negative_index_read_all(self, mp4_file):
        """Test that negative frame index raises IndexError in read_all mode."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path), read_all=True)

        with pytest.raises(IndexError):
            reader.get_frame(-1)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file.

        BUG 4: MP4Reader.__del__ crashes with AttributeError when __init__ raises,
        because self.cap is never initialized. The destructor should handle this gracefully.
        """
        with pytest.raises(FileNotFoundError):
            MP4Reader("/nonexistent/path/video.mp4")

    def test_read_all_mode(self, mp4_file):
        """Test reading all frames into memory."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path), read_all=True)

        assert reader.frame_count == 5
        assert reader.cap is None  # Capture released after reading all

        frame = reader.get_frame(0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)

    def test_read_all_mode_all_frames_accessible(self, mp4_file):
        """Test that all frames are accessible in read_all mode."""
        mp4_path, _ = mp4_file
        reader = MP4Reader(str(mp4_path), read_all=True)

        for i in range(5):
            frame = reader.get_frame(i)
            assert frame is not None
            assert frame.shape == (64, 64, 3)


class TestMP4Roundtrip:
    """Test MP4 write/read roundtrip."""

    def test_roundtrip_preserves_frame_count(self, tmp_path):
        """Test that write/read roundtrip preserves frame count."""
        output_path = tmp_path / "test.mp4"
        n_frames = 10

        writer = MP4Writer(output_path, fps=30.0)
        for _ in range(n_frames):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        reader = MP4Reader(str(output_path))
        assert reader.frame_count == n_frames

    def test_roundtrip_preserves_frame_shape(self, tmp_path):
        """Test that write/read roundtrip preserves frame dimensions."""
        output_path = tmp_path / "test.mp4"
        height, width = 128, 256

        writer = MP4Writer(output_path, fps=30.0)
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()

        reader = MP4Reader(str(output_path))
        assert reader.width == width
        assert reader.height == height

        read_frame = reader.get_frame(0)
        assert read_frame.shape == (height, width, 3)

    def test_roundtrip_approximate_values(self, tmp_path):
        """Test that write/read roundtrip preserves values approximately (lossy codec)."""
        output_path = tmp_path / "test.mp4"

        writer = MP4Writer(output_path, fps=30.0)
        original_frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        writer.write_frame(original_frame)
        writer.close()

        reader = MP4Reader(str(output_path))
        read_frame = reader.get_frame(0)

        # MP4 codec is lossy, allow tolerance
        assert np.mean(np.abs(read_frame.astype(float) - original_frame.astype(float))) < 30.0

    def test_cached_reader(self, tmp_path):
        """Test that get_mp4_reader_from_path returns cached readers."""
        output_path = tmp_path / "cached_test.mp4"
        writer = MP4Writer(output_path, fps=30.0)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()

        # Clear LRU cache to avoid cross-test interference
        get_mp4_reader_from_path.cache_clear()

        reader1 = get_mp4_reader_from_path(str(output_path))
        reader2 = get_mp4_reader_from_path(str(output_path))
        assert reader1 is reader2

        get_mp4_reader_from_path.cache_clear()
