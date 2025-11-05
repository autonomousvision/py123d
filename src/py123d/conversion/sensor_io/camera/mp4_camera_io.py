# TODO: add method of handling camera mp4 io
def load_image_from_mp4_file() -> None:
    raise NotImplementedError


from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class MP4Writer:
    """Write images sequentially to an MP4 video file."""

    def __init__(self, output_path: Union[str, Path], fps: float = 30.0, codec: str = "mp4v"):
        """
        Initialize MP4 writer.

        Args:
            output_path: Path to output MP4 file
            fps: Frames per second
            codec: Video codec ('mp4v', 'avc1', 'h264')
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray) -> int:
        """
        Write a single frame to the video.

        Args:
            frame: Image as numpy array (RGB format)
        """
        frame_idx = int(self.frame_count)
        if self.writer is None:
            # Initialize writer with first frame's dimensions
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(f"Frame size {frame.shape[:2][::-1]} doesn't match " f"video size {self.frame_size}")

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
        self.frame_count += 1
        return frame_idx

    def close(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class MP4Reader:
    """Read MP4 video with random frame access."""

    def __init__(self, video_path: Union[str, Path], read_all: bool = False):
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
        self.read_all = read_all

        if read_all:
            self.frames = []
            for _ in range(self.frame_count):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
            self.cap.release()
            self.cap = None

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by index.

        Args:
            frame_index: Zero-based frame index

        Returns:
            Frame as numpy array (RGB format) or None if invalid index
        """

        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range " f"[0, {len(self.frames)})")

        if self.read_all:
            return self.frames[frame_index]

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Convert BGR to RGB for sane convention
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame if ret else None

    def __getitem__(self, index: int) -> np.ndarray:
        """Allow indexing like reader[10]"""
        return self.get_frame(index)


@lru_cache(maxsize=64)
def get_mp4_reader_from_path(mp4_path: str) -> MP4Reader:
    return MP4Reader(mp4_path, read_all=False)
