from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

# Lazy-initialized TurboJPEG instance (created on first use when scaling is requested).
_turbojpeg_instance = None


def _get_turbojpeg():
    """Return a module-level TurboJPEG instance, creating it on first call."""
    global _turbojpeg_instance
    if _turbojpeg_instance is None:
        from turbojpeg import TurboJPEG

        _turbojpeg_instance = TurboJPEG()
    return _turbojpeg_instance


def is_jpeg_binary(jpeg_binary: bytes) -> bool:
    """Check if the given binary data represents a JPEG image.

    :param jpeg_binary: The binary data to check.
    :return: True if the binary data is a JPEG image, False otherwise.
    """
    SOI_MARKER = b"\xff\xd8"  # Start Of Image
    EOI_MARKER = b"\xff\xd9"  # End Of Image

    return jpeg_binary.startswith(SOI_MARKER) and jpeg_binary.endswith(EOI_MARKER)


def encode_image_as_jpeg_binary(image: npt.NDArray[np.uint8]) -> bytes:
    """Encodes a numpy RGB image as JPEG binary."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode(".jpg", image)
    jpeg_binary = encoded_img.tobytes()
    return jpeg_binary


def decode_image_from_jpeg_binary(
    jpeg_binary: bytes,
    scaling_factor: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    """Decodes a numpy RGB image from JPEG binary.

    :param jpeg_binary: The JPEG binary data to decode.
    :param scaling_factor: Optional (numerator, denominator) tuple for downscaling during decode,
        e.g. (1, 2) for half size, (1, 4) for quarter size. Requires the ``turbojpeg`` package.
    """
    if scaling_factor is not None:
        from turbojpeg import TJPF_RGB

        tj = _get_turbojpeg()
        image = tj.decode(jpeg_binary, pixel_format=TJPF_RGB, scaling_factor=scaling_factor)
        return image

    image = cv2.imdecode(np.frombuffer(jpeg_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_jpeg_binary_from_jpeg_file(jpeg_path: Path) -> bytes:
    """Loads JPEG binary data from a JPEG file."""
    with open(jpeg_path, "rb") as f:
        jpeg_binary = f.read()
    return jpeg_binary


def load_image_from_jpeg_file(
    jpeg_path: Path,
    scaling_factor: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    """Loads a numpy RGB image from a JPEG file.

    :param jpeg_path: Path to the JPEG file.
    :param scaling_factor: Optional (numerator, denominator) tuple for downscaling during decode,
        e.g. (1, 2) for half size, (1, 4) for quarter size. Requires the ``turbojpeg`` package.
    """
    if scaling_factor is not None:
        jpeg_binary = load_jpeg_binary_from_jpeg_file(jpeg_path)
        return decode_image_from_jpeg_binary(jpeg_binary, scaling_factor=scaling_factor)

    image = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
