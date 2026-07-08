from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.base_camera import (
    BaseCameraMetadata,
    CameraChannelType,
    CameraID,
    CameraModel,
    camera_metadata_from_dict,
)
from py123d.geometry.pose import PoseSE3

DEPTH_BITS = (8, 16)
"""Supported PNG bit depths. ``8`` yields an 8-bit grayscale PNG, ``16`` a 16-bit one."""

_DEPTH_DTYPES: Dict[int, np.dtype] = {8: np.dtype(np.uint8), 16: np.dtype(np.uint16)}


class DepthCameraMetadata(BaseCameraMetadata):
    """Metadata for a per-pixel depth camera stream.

    A depth camera is pixel-aligned to a regular RGB :class:`~py123d.datatypes.Camera`: it shares that
    camera's id, projection model, intrinsics, and extrinsics. This metadata therefore *composes* the
    sibling camera's :class:`BaseCameraMetadata` for all geometry and additionally records **how the
    continuous metric depth was quantized** into the stored integer raster — analogous to how
    :class:`~py123d.datatypes.segmentation_camera.SegmentationCameraMetadata` records which label
    taxonomy its integer class ids use.

    Its :attr:`channel_type` is always :attr:`CameraChannelType.DEPTH`
    (:attr:`ModalityType.CAMERA_DEPTH`, file ``camera_depth.<camera_id>.arrow``), written to its own
    Arrow file, never colliding with the RGB ``camera.<camera_id>`` that shares its :attr:`camera_id`.

    **Storage contract.** Depth is continuous, so — unlike a segmentation class-id map — it must be
    quantized before it can be stored as a lossless integer PNG. The encoding is a clip followed by a
    linear rescale onto the full integer range::

        raw = round(clip(depth_m, 0, max_depth) / max_depth * max_raw)     # encode
        depth_m = raw / max_raw * max_depth                                 # decode

    where ``max_raw = 2 ** depth_bits - 1``. Two consequences worth internalizing:

    * **The far plane is a hard clip, not a sentinel.** Anything beyond :attr:`max_depth` saturates to
      ``max_raw`` and decodes back as exactly :attr:`max_depth`. A CARLA sky pixel at 1000 m and a wall
      at :attr:`max_depth` are indistinguishable after encoding. Choose :attr:`max_depth` accordingly.
    * **``0`` means zero metres, not "no measurement".** There is no invalid sentinel: the full integer
      range encodes depth. This suits simulators (CARLA renders a finite depth for every pixel); a
      real-world sensor with dropouts needs its invalid mask stored separately.

    :attr:`depth_bits` therefore trades resolution against file size, and :attr:`max_depth` trades
    range against resolution. The worst-case round-trip error is half a quantization step,
    :attr:`depth_resolution` / 2:

    ===========  =============  ==================  ======================
    depth_bits   max_depth      depth_resolution    max round-trip error
    ===========  =============  ==================  ======================
    8            50 m           196 mm              98 mm
    16           96 m           1.46 mm             0.73 mm
    16           1024 m         15.6 mm             7.8 mm
    ===========  =============  ==================  ======================
    """

    __slots__ = ("_camera_metadata", "_max_depth", "_depth_bits")

    def __init__(
        self,
        camera_metadata: BaseCameraMetadata,
        max_depth: float,
        depth_bits: int = 16,
    ) -> None:
        """Initialize a :class:`DepthCameraMetadata`.

        :param camera_metadata: The sibling RGB camera metadata providing geometry (id, model,
            intrinsics, extrinsics, resolution).
        :param max_depth: The far clipping plane, in metres. Depth at or beyond this saturates to the
            largest storable integer. This is the *only* range knob; ``depth_scale`` is derived from it.
        :param depth_bits: ``8`` or ``16``. Selects the stored dtype (``uint8``/``uint16``) and hence
            the PNG bit depth. Defaults to ``16``.
        """
        assert depth_bits in DEPTH_BITS, f"depth_bits must be one of {DEPTH_BITS}, got {depth_bits}."
        assert max_depth > 0.0, f"max_depth must be positive, got {max_depth}."
        self._camera_metadata = camera_metadata
        self._max_depth = float(max_depth)
        self._depth_bits = int(depth_bits)

    @property
    def camera_metadata(self) -> BaseCameraMetadata:
        """The sibling camera metadata that provides this stream's geometry."""
        return self._camera_metadata

    @property
    def max_depth(self) -> float:
        """The far clipping plane in metres; depth at or beyond this saturates to :attr:`max_raw`."""
        return self._max_depth

    @property
    def depth_bits(self) -> int:
        """``8`` or ``16``: the bit depth of the stored integer raster (and of the PNG)."""
        return self._depth_bits

    @property
    def max_raw(self) -> int:
        """The largest storable integer, ``2 ** depth_bits - 1`` (i.e. ``255`` or ``65535``)."""
        return (1 << self._depth_bits) - 1

    @property
    def depth_dtype(self) -> np.dtype:
        """The numpy dtype of the stored raster: ``uint8`` or ``uint16``."""
        return _DEPTH_DTYPES[self._depth_bits]

    @property
    def depth_resolution(self) -> float:
        """Metres per integer unit, ``max_depth / max_raw``. One quantization step."""
        return self._max_depth / self.max_raw

    @property
    def channel_type(self) -> CameraChannelType:
        """Always :attr:`CameraChannelType.DEPTH`."""
        return CameraChannelType.DEPTH

    # ------------------------------------------------------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------------------------------------------------------

    def encode_depth(self, depth: npt.NDArray[np.floating]) -> npt.NDArray[np.unsignedinteger]:
        """Quantize a metric depth map ``(H, W)`` in metres into the stored integer raster.

        Clips to ``[0, max_depth]``, rescales linearly onto ``[0, max_raw]``, and rounds to nearest
        (rather than truncating, which would bias every pixel downward by half a step on average).
        Non-finite values (``NaN``/``inf``, e.g. from an unrendered pixel) clamp to the far plane.

        :param depth: A 2D ``(H, W)`` float array of depths in metres.
        :return: A 2D ``(H, W)`` array of :attr:`depth_dtype`.
        """
        assert depth.ndim == 2, f"Depth map must be a single-channel (H, W) array, got shape {depth.shape}."
        # nan -> max_depth (an unrendered / infinitely distant pixel is "as far as we can say"), and
        # +-inf -> the clip bounds. Done before the clip so NaN doesn't propagate through it.
        depth_m = np.nan_to_num(
            np.asarray(depth, dtype=np.float64),
            nan=self._max_depth,
            posinf=self._max_depth,
            neginf=0.0,
        )
        depth_m = np.clip(depth_m, 0.0, self._max_depth)
        raw = np.round(depth_m / self._max_depth * self.max_raw)
        return raw.astype(self.depth_dtype)

    def decode_depth(self, raw: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
        """Dequantize a stored integer raster back to metric depth in metres.

        The inverse of :meth:`encode_depth`, up to the quantization error (at most
        :attr:`depth_resolution` / 2). Note that pixels which saturated on encode decode to exactly
        :attr:`max_depth`, not to their true distance.

        :param raw: A 2D ``(H, W)`` integer array as returned by :meth:`encode_depth`.
        :return: A 2D ``(H, W)`` float32 array of depths in metres, in ``[0, max_depth]``.
        """
        # A single multiply by the (float64-computed, then narrowed) step, rather than a float32
        # divide-then-multiply: one rounding instead of two, so the result stays within half a
        # float32 ulp of the exact dequantized value.
        return np.asarray(raw, dtype=np.float32) * np.float32(self.depth_resolution)

    # ------------------------------------------------------------------------------------------------------------------
    # Geometry delegated to the sibling camera metadata
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def camera_model(self) -> CameraModel:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_model

    @property
    def camera_id(self) -> CameraID:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_id

    @property
    def camera_name(self) -> str:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_name

    @property
    def camera_to_imu_se3(self) -> PoseSE3:
        """Inherited, see superclass."""
        return self._camera_metadata.camera_to_imu_se3

    @property
    def width(self) -> int:
        """Inherited, see superclass."""
        return self._camera_metadata.width

    @property
    def height(self) -> int:
        """Inherited, see superclass."""
        return self._camera_metadata.height

    def project_to_image(
        self,
        points_cam: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Inherited, see superclass."""
        return self._camera_metadata.project_to_image(points_cam)

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata, embedding the sibling camera and the quantization contract."""
        return {
            "camera_metadata": self._camera_metadata.to_dict(),
            "max_depth": self._max_depth,
            "depth_bits": self._depth_bits,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> DepthCameraMetadata:
        """Construct a :class:`DepthCameraMetadata` from a dictionary."""
        return cls(
            camera_metadata=camera_metadata_from_dict(data_dict["camera_metadata"]),
            max_depth=data_dict["max_depth"],
            depth_bits=data_dict["depth_bits"],
        )

    def __repr__(self) -> str:
        return (
            f"DepthCameraMetadata(camera_id={self.camera_id}, "
            f"max_depth={self._max_depth}, depth_bits={self._depth_bits})"
        )

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """The camera id, so the stream sits at ``camera_depth.<camera_id>``."""
        return self.camera_id


def colorize_depth_map(
    depth_raw: npt.NDArray[np.integer],
    max_raw: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    """Colorize a stored integer depth raster into a ``(H, W, 3)`` uint8 RGB image for display.

    Normalizes against the *dtype's* full range rather than the frame's min/max, so the colour of a
    given distance is stable across frames (a per-frame min/max would make the palette flicker as
    objects enter and leave the view). Near is warm, far is cool.

    :param depth_raw: A 2D ``(H, W)`` integer depth raster (``uint8`` or ``uint16``).
    :param max_raw: The largest storable integer. Defaults to the maximum of ``depth_raw.dtype``.
    :return: A ``(H, W, 3)`` uint8 RGB image.
    """
    import cv2

    if max_raw is None:
        max_raw = int(np.iinfo(depth_raw.dtype).max)
    normalized = np.clip(np.asarray(depth_raw, dtype=np.float64) / max_raw, 0.0, 1.0)
    # TURBO is perceptually uniform and reads naturally as a depth ramp. cv2 colormaps emit BGR.
    colored_bgr = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
