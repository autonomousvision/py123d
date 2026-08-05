from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import numpy.typing as npt

# Decoder replacing the built-in one, set by a reader that has a faster PNG library
# available. None keeps the OpenCV path.
_png_decoder: Optional[Callable[[bytes], npt.NDArray]] = None


def set_png_decoder(decoder: Optional[Callable[[bytes], npt.NDArray]]) -> None:
    """Use ``decoder`` for full-size PNG decoding, or None for the built-in one.

    Applies to both image and label-map decoding. An image decoder must return an HWC
    RGB array; a label map comes back as the 2D array the file holds.

    :param decoder: Callable taking PNG binary and returning the decoded array.
    """
    global _png_decoder
    _png_decoder = decoder


def is_png_binary(png_binary: bytes) -> bool:
    """Check if the given binary data represents a PNG image.

    :param png_binary: The binary data to check.
    :return: True if the binary data is a PNG image, False otherwise.
    """
    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"  # PNG file signature

    return png_binary.startswith(PNG_SIGNATURE)


def encode_image_as_png_binary(image: npt.NDArray[np.uint8]) -> bytes:
    """Encodes a numpy RGB image as PNG binary."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # type: ignore
    _, encoded_img = cv2.imencode(".png", image)
    png_binary = encoded_img.tobytes()
    return png_binary


def decode_image_from_png_binary(png_binary: bytes, scale: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """Decodes a numpy image from PNG binary.

    :param png_binary: The PNG binary data to decode.
    :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
    """
    if _png_decoder is not None:
        image = _png_decoder(png_binary)
    else:
        image = cv2.imdecode(np.frombuffer(png_binary, np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale is not None and scale > 1:
        new_h = image.shape[0] // scale
        new_w = image.shape[1] // scale
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image  # type: ignore[return-value]


def load_png_binary_from_png_file(png_path: Path) -> bytes:
    """Loads PNG binary data from a PNG file."""
    with open(png_path, "rb") as f:
        png_binary = f.read()
    return png_binary


def load_image_from_png_file(png_path: Path) -> npt.NDArray[np.uint8]:
    """Loads a numpy image from a PNG file."""
    image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------------------------------
# Single-channel integer label maps (semantic segmentation)
# ------------------------------------------------------------------------------------------------------------------


def encode_label_map_as_png_binary(label_map: npt.NDArray[np.integer]) -> bytes:
    """Encodes a single-channel integer label map (H, W) as lossless PNG binary.

    Unlike :func:`encode_image_as_png_binary`, no color conversion is applied: the raw class ids
    are preserved exactly. ``uint8`` maps become 8-bit PNGs and ``uint16`` maps 16-bit PNGs.

    :param label_map: A 2D (H, W) array of integer class ids (uint8 or uint16).
    :return: Lossless PNG binary preserving the integer values.
    """
    assert label_map.ndim == 2, f"Label map must be a single-channel (H, W) array, got shape {label_map.shape}."
    _, encoded_label_map = cv2.imencode(".png", label_map)
    return encoded_label_map.tobytes()


def decode_label_map_from_png_binary(png_binary: bytes, scale: Optional[int] = None) -> npt.NDArray[np.integer]:
    """Decodes a single-channel integer label map from PNG binary, preserving class ids exactly.

    :param png_binary: The PNG binary data to decode.
    :param scale: Optional downscale denominator. Nearest-neighbour is used so class ids are never blended.
    :return: A 2D (H, W) array of integer class ids with the original dtype (uint8/uint16).
    """
    if _png_decoder is not None:
        label_map = _png_decoder(png_binary)
    else:
        label_map = cv2.imdecode(np.frombuffer(png_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    if scale is not None and scale > 1:
        new_h = label_map.shape[0] // scale
        new_w = label_map.shape[1] // scale
        label_map = cv2.resize(label_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return label_map  # type: ignore[return-value]
