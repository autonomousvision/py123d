from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def is_jpeg_binary(jpeg_binary: bytes) -> bool:
    """Check if the given binary data represents a JPEG image.

    :param jpeg_binary: The binary data to check.
    :return: True if the binary data is a JPEG image, False otherwise.
    """
    SOI_MARKER = b"\xff\xd8"  # Start Of Image
    EOI_MARKER = b"\xff\xd9"  # End Of Image

    return jpeg_binary.startswith(SOI_MARKER) and jpeg_binary.endswith(EOI_MARKER)


def encode_image_as_jpeg_binary(image: npt.NDArray[np.uint8]) -> bytes:
    _, encoded_img = cv2.imencode(".jpg", image)
    jpeg_binary = encoded_img.tobytes()
    return jpeg_binary


def decode_image_from_jpeg_binary(jpeg_binary: bytes) -> npt.NDArray[np.uint8]:
    image = cv2.imdecode(np.frombuffer(jpeg_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_jpeg_binary_from_jpeg_file(jpeg_path: Path) -> bytes:
    with open(jpeg_path, "rb") as f:
        jpeg_binary = f.read()
    return jpeg_binary


def load_image_from_jpeg_file(jpeg_path: Path) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
