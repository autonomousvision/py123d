"""On-the-fly display ISP for raw-stored camera images.

Datasets can attach a display-ISP block to their camera metadata (see
``FThetaCameraMetadata.isp``): black/white level, a 3x3 color correction matrix,
per-channel tone curves given as sparse control points, and optionally ``storage_gamma``
(default 2.2) — the exponent applied to stored values to reach the domain the block's
parameters are defined in (1.0 for blocks fitted directly on the stored values). The
stored images carry no color processing; this module applies the color transform at
display time.

The pipeline is four OpenCV calls, each SIMD-vectorized and multi-threaded:
``cv2.LUT`` (decode gamma + black/white normalization into a uint16 linear domain),
``cv2.transform`` (color correction matrix, saturating), ``cv2.convertScaleAbs``
(saturating down-conversion to uint8) and ``cv2.LUT`` (per-channel tone curves).
The uint16 intermediate keeps quantization error at or below 2/255, matching the
error budget of the sparse tone-curve representation itself.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

# Default exponent for ISP blocks that do not declare storage_gamma (sRGB-style). Kesai
# blocks declare 1.0: their parameters are fitted directly on the stored values.
_DEFAULT_STORAGE_GAMMA: float = 2.2

# LUTs per distinct ISP block, keyed by a content hash.
_LUT_CACHE: Dict[int, Tuple[npt.NDArray[np.uint16], npt.NDArray[np.float32], npt.NDArray[np.uint8]]] = {}


def _isp_cache_key(isp: Dict[str, Any]) -> int:
    tone = isp["tone_curve"]
    return hash(
        (
            float(isp.get("storage_gamma", _DEFAULT_STORAGE_GAMMA)),
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
) -> Tuple[npt.NDArray[np.uint16], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Expand an ISP block into the lookup structures used per frame."""
    storage_gamma = float(isp.get("storage_gamma", _DEFAULT_STORAGE_GAMMA))
    black = float(isp["black_level"])
    white = float(isp["white_level"])

    # uint8 pixel -> normalized linear value (decode storage gamma, remove pedestal),
    # quantized to uint16 so the color matrix runs in an integer saturating domain.
    encoded = np.arange(256, dtype=np.float64) / 255.0
    linear = np.clip((encoded**storage_gamma - black) / (white - black), 0.0, 1.0)
    input_lut = np.round(linear * 65535.0).astype(np.uint16).reshape(1, 256, 1)

    ccm = np.asarray(isp["ccm"], dtype=np.float32)

    # Sparse monotone control points -> dense per-channel uint8 tone table, merged into
    # the (1, 256, 3) layout cv2.LUT expects for per-channel lookups.
    tone = isp["tone_curve"]
    knots = np.asarray(tone["x"], dtype=np.float64)
    dense_x = np.linspace(0.0, 1.0, 256)
    tone_lut = np.stack(
        [
            np.clip(PchipInterpolator(knots, np.asarray(tone[name], dtype=np.float64))(dense_x) * 255.0, 0, 255).astype(
                np.uint8
            )
            for name in ("red", "green", "blue")
        ],
        axis=-1,
    ).reshape(1, 256, 3)
    return input_lut, ccm, tone_lut


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
    input_lut, ccm, tone_lut = luts

    linear = cv2.LUT(np.ascontiguousarray(image), input_lut)
    corrected = cv2.transform(linear, ccm)
    return cv2.LUT(cv2.convertScaleAbs(corrected, alpha=1.0 / 256.0), tone_lut)
