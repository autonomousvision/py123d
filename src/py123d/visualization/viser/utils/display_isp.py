"""On-the-fly display ISP for raw-stored camera images.

Datasets can attach a display-ISP block to their camera metadata (see
``FThetaCameraMetadata.isp``): black/white level, a 3x3 color correction matrix and
per-channel tone curves given as sparse control points. The stored images are raw
(gamma-encoded sensor data); this module applies the color transform at display time.

The pipeline is reduced to three vectorized steps so it stays fast enough for per-frame
use: one 256-entry gather (decode gamma + black/white normalization), one (N, 3) @ (3, 3)
matrix product, and one 4096-entry gather per channel (tone curve, quantized to uint8).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

# Stored images are sRGB-style gamma encoded with this exponent.
_STORAGE_GAMMA: float = 2.2

# Resolution of the dense tone lookup table expanded from the sparse control points.
_TONE_LUT_SIZE: int = 4096

# LUTs per distinct ISP block, keyed by a content hash.
_LUT_CACHE: Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uint8]]] = {}


def _isp_cache_key(isp: Dict[str, Any]) -> int:
    tone = isp["tone_curve"]
    return hash(
        (
            float(isp["black_level"]),
            float(isp["white_level"]),
            tuple(tuple(row) for row in isp["ccm"]),
            tuple(tone["x"]),
            tuple(tone["red"]),
            tuple(tone["green"]),
            tuple(tone["blue"]),
        )
    )


def _build_luts(
    isp: Dict[str, Any],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Expand an ISP block into the three lookup structures used per frame."""
    black = float(isp["black_level"])
    white = float(isp["white_level"])

    # uint8 pixel -> normalized linear value (decode storage gamma, remove pedestal).
    encoded = np.arange(256, dtype=np.float64) / 255.0
    linear = encoded**_STORAGE_GAMMA
    input_lut = np.clip((linear - black) / (white - black), 0.0, 1.0).astype(np.float32)

    ccm_t = np.asarray(isp["ccm"], dtype=np.float32).T

    # Sparse monotone control points -> dense per-channel uint8 tone table.
    tone = isp["tone_curve"]
    knots = np.asarray(tone["x"], dtype=np.float64)
    dense_x = np.linspace(0.0, 1.0, _TONE_LUT_SIZE)
    tone_lut = np.stack(
        [
            np.clip(PchipInterpolator(knots, np.asarray(tone[name], dtype=np.float64))(dense_x) * 255.0, 0, 255).astype(
                np.uint8
            )
            for name in ("red", "green", "blue")
        ]
    )
    return input_lut, ccm_t, tone_lut


def apply_display_isp(image: npt.NDArray[np.uint8], isp: Optional[Dict[str, Any]]) -> npt.NDArray[np.uint8]:
    """Apply a display-ISP block to a stored raw RGB image.

    :param image: (H, W, 3) uint8 RGB image as stored in the dataset.
    :param isp: The ISP block from the camera metadata, or None for a no-op.
    :return: The corrected (H, W, 3) uint8 RGB image (the input when isp is None).
    """
    if isp is None or image is None or image.ndim != 3 or image.shape[2] != 3:
        return image

    key = _isp_cache_key(isp)
    luts = _LUT_CACHE.get(key)
    if luts is None:
        luts = _build_luts(isp)
        _LUT_CACHE[key] = luts
    input_lut, ccm_t, tone_lut = luts

    x = input_lut[image]
    x = np.clip(x @ ccm_t, 0.0, 1.0)
    indices = (x * (_TONE_LUT_SIZE - 1)).astype(np.uint16)
    return np.stack([tone_lut[channel][indices[..., channel]] for channel in range(3)], axis=-1)
