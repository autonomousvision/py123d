# TODO: add method of handling camera mp4 io
def load_image_from_mp4_file() -> None:
    raise NotImplementedError


from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class MP4Writer:
    """Write images sequentially to an MP4 video file."""

    def __init__(self, output_path: str, fps: float = 30.0, codec: str = "mp4v"):
        """
        Initialize MP4 writer.

        Args:
            output_path: Path to output MP4 file
            fps: Frames per second
            codec: Video codec ('mp4v', 'avc1', 'h264')
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to the video.

        Args:
            frame: Image as numpy array (BGR format)
        """
        if self.writer is None:
            # Initialize writer with first frame's dimensions
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(f"Frame size {frame.shape[:2][::-1]} doesn't match " f"video size {self.frame_size}")

        self.writer.write(frame)
        self.frame_count += 1

    def close(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MP4Reader:
    """Read MP4 video with random frame access."""

    def __init__(self, video_path: str):
        """
        Initialize MP4 reader.

        Args:
            video_path: Path to MP4 file
        """
        self.video_path = video_path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by index.

        Args:
            frame_index: Zero-based frame index

        Returns:
            Frame as numpy array (BGR format) or None if invalid index
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range " f"[0, {self.frame_count})")

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()

        return frame if ret else None

    def read_sequential(self) -> Optional[np.ndarray]:
        """
        Read next frame sequentially.

        Returns:
            Frame as numpy array or None if end of video
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def reset(self):
        """Reset to beginning of video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.frame_count

    def __getitem__(self, index: int) -> np.ndarray:
        """Allow indexing like reader[10]"""
        return self.get_frame(index)


# Example usage
if __name__ == "__main__":
    # Create sample video
    print("Creating sample video...")
    with MP4Writer("output.mp4", fps=30.0) as writer:
        for i in range(100):
            # Create colored frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            color = int(255 * i / 100)
            frame[:, :] = (color, 255 - color, 128)

            # Add frame number text
            cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            writer.write_frame(frame)

    print(f"Video created with {writer.frame_count} frames")

    # Read video with indexing
    print("\nReading video with random access...")
    with MP4Reader("output.mp4") as reader:
        print(f"Video info: {len(reader)} frames, " f"{reader.width}x{reader.height}, {reader.fps} fps")

        # Read specific frames
        frames_to_read = [0, 25, 50, 75, 99]
        for idx in frames_to_read:
            frame = reader[idx]
            if frame is not None:
                print(f"Successfully read frame {idx}")
            else:
                print(f"Failed to read frame {idx}")

        # Sequential reading example
        print("\nReading first 5 frames sequentially...")
        reader.reset()
        for i in range(5):
            frame = reader.read_sequential()
            if frame is not None:
                print(f"Read sequential frame {i}")
