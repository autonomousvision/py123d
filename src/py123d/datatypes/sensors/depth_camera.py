from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Union

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

DepthTransform = Literal["linear", "inverse"]
"""How metric depth is mapped onto the integer range before quantization.

* ``"linear"`` spreads the codes uniformly over metres in ``[min_depth, max_depth]`` — equal metric
  precision everywhere.
* ``"inverse"`` spreads them uniformly over *inverse* depth (disparity): fine resolution near the
  camera, coarse far away (absolute error grows ~quadratically with depth). This concentrates
  precision where 3D reconstruction needs it and matches how stereo/disparity sensors quantize. It
  requires a positive ``min_depth`` (``1 / depth`` diverges at zero).
"""
DEPTH_TRANSFORMS = ("linear", "inverse")

DepthType = Literal["z_depth", "ray_distance"]
"""What the stored scalar measures for each pixel.

* ``"z_depth"`` — the coordinate along the optical axis (``Z`` in the camera frame): the perpendicular
  distance to the image plane. This is what rasterizer depth buffers and most RGB-D sensors report,
  and what :meth:`project_to_image` returns, so it is the default.
* ``"ray_distance"`` — the euclidean range from the camera centre, ``sqrt(X**2 + Y**2 + Z**2)``. Native
  to raw range sensors and some panoramic renders.

This flag is a provenance annotation: it does not affect quantization (which sees only a per-pixel
scalar), but any code that unprojects the depth map back to 3D must branch on it.
"""
DEPTH_TYPES = ("z_depth", "ray_distance")


class DepthCameraMetadata(BaseCameraMetadata):
    """Metadata for a per-pixel depth camera stream.

    A depth camera is pixel-aligned to a sibling RGB camera: it shares that camera's id, projection
    model, intrinsics, and extrinsics, and composes its ``BaseCameraMetadata`` for all geometry. In
    addition, it records how the continuous metric depth was quantized into the stored integer raster.

    Its ``channel_type`` is always ``DEPTH`` (modality ``CAMERA_DEPTH``, file
    ``camera_depth.<camera_id>.arrow``), a separate stream from the RGB ``camera.<camera_id>.arrow`` of
    the same ``camera_id``.

    **Storage contract.** Depth is continuous, so it is quantized to an unsigned integer raster
    (``uint8`` or ``uint16``) and stored as a lossless PNG. Quantization clips to ``[min_depth,
    max_depth]``, maps onto the unit interval (linearly in depth, or in inverse depth — see
    ``depth_transform``), and rounds onto the integer codes ``[offset, max_raw]``::

        t   = to_unit(clip(depth_m, min_depth, max_depth))   # depth -> [0, 1]  (transform-dependent)
        raw = offset + round(t * (max_raw - offset))         # encode
        # decode inverts to_unit on (raw - offset) / (max_raw - offset)

    where ``max_raw = 2 ** depth_bits - 1`` and ``offset`` is ``1`` when ``has_invalid`` reserves code
    ``0`` (else ``0``). Points to keep in mind:

    * ``max_depth`` clips the far range. Anything at or beyond ``max_depth`` saturates to ``max_raw``
      and decodes back as exactly ``max_depth``, so a sky pixel at 1000 m and a wall at ``max_depth``
      are indistinguishable once encoded.
    * **Invalid pixels.** With ``has_invalid=False`` (the default) there is no sentinel: the full
      integer range encodes depth, ``0`` means zero metres, and non-finite inputs clamp to the range.
      This suits simulators, which render a finite depth for every pixel. With ``has_invalid=True``,
      code ``0`` is reserved for "no measurement": any non-finite or non-positive input encodes to
      ``0`` and decodes back to ``NaN``, and valid depth uses ``[1, max_raw]``. Set this for real
      sensors (lidar-projected, ToF, stereo) that have dropouts.
    * ``depth_transform`` chooses linear vs. inverse-depth spacing; ``depth_type`` records whether the
      scalar is planar z-depth or euclidean range (see those knobs).

    ``depth_bits`` trades resolution against file size; ``max_depth`` trades range against resolution.
    For ``linear`` the worst-case error is half a quantization step (``depth_resolution`` / 2):

    ===========  =============  ==================  ==========
    depth_bits   max_depth      depth_resolution    max error
    ===========  =============  ==================  ==========
    8            50 m           196 mm              98 mm
    16           96 m           1.46 mm             0.73 mm
    16           1024 m         15.6 mm             7.8 mm
    ===========  =============  ==================  ==========
    """

    __slots__ = (
        "_camera_metadata",
        "_max_depth",
        "_depth_bits",
        "_depth_transform",
        "_min_depth",
        "_has_invalid",
        "_depth_type",
    )

    def __init__(
        self,
        camera_metadata: BaseCameraMetadata,
        max_depth: float,
        depth_bits: int = 16,
        depth_transform: DepthTransform = "linear",
        min_depth: float = 0.0,
        has_invalid: bool = False,
        depth_type: DepthType = "z_depth",
    ) -> None:
        """Initialize a ``DepthCameraMetadata``.

        :param camera_metadata: The sibling RGB camera metadata providing geometry (id, model,
            intrinsics, extrinsics, resolution).
        :param max_depth: The far clipping plane, in metres. Depth at or beyond this saturates to the
            largest storable integer.
        :param depth_bits: ``8`` or ``16``. Selects the stored dtype (``uint8``/``uint16``) and hence
            the PNG bit depth. Defaults to ``16``.
        :param depth_transform: ``"linear"`` (uniform metric spacing, the default) or ``"inverse"``
            (uniform disparity spacing — fine near, coarse far). ``"inverse"`` requires a positive
            ``min_depth``.
        :param min_depth: The near clipping plane in metres. Required (``> 0``) for the ``"inverse"``
            transform, where it bounds the disparity range. For ``"linear"`` it is an optional floor and
            defaults to ``0.0`` (i.e. no near clip), preserving the historic behaviour.
        :param has_invalid: If ``True``, reserve code ``0`` as a "no measurement" sentinel and encode
            valid depth into ``[1, max_raw]``; invalid pixels decode to ``NaN``. Defaults to ``False``.
        :param depth_type: ``"z_depth"`` (planar, along the optical axis — the default) or
            ``"ray_distance"`` (euclidean range from the camera centre). A provenance flag only; it does
            not affect quantization.
        """
        assert depth_bits in DEPTH_BITS, f"depth_bits must be one of {DEPTH_BITS}, got {depth_bits}."
        assert max_depth > 0.0, f"max_depth must be positive, got {max_depth}."
        assert depth_transform in DEPTH_TRANSFORMS, (
            f"depth_transform must be one of {DEPTH_TRANSFORMS}, got {depth_transform!r}."
        )
        assert depth_type in DEPTH_TYPES, f"depth_type must be one of {DEPTH_TYPES}, got {depth_type!r}."
        assert 0.0 <= min_depth < max_depth, f"min_depth must be in [0, max_depth), got {min_depth}."
        if depth_transform == "inverse":
            assert min_depth > 0.0, "The 'inverse' transform requires a positive min_depth (1 / depth diverges at 0)."
        self._camera_metadata = camera_metadata
        self._max_depth = float(max_depth)
        self._depth_bits = int(depth_bits)
        self._depth_transform: DepthTransform = depth_transform
        self._min_depth = float(min_depth)
        self._has_invalid = bool(has_invalid)
        self._depth_type: DepthType = depth_type

    @property
    def camera_metadata(self) -> BaseCameraMetadata:
        """The sibling camera metadata that provides this stream's geometry."""
        return self._camera_metadata

    @property
    def max_depth(self) -> float:
        """The far clipping plane in metres; depth at or beyond this saturates to ``max_raw``."""
        return self._max_depth

    @property
    def min_depth(self) -> float:
        """The near clipping plane in metres. ``0`` (no near clip) unless set, and required for ``inverse``."""
        return self._min_depth

    @property
    def depth_bits(self) -> int:
        """``8`` or ``16``: the bit depth of the stored integer raster (and of the PNG)."""
        return self._depth_bits

    @property
    def depth_transform(self) -> DepthTransform:
        """``"linear"`` or ``"inverse"``: how metric depth is spaced across the integer codes."""
        return self._depth_transform

    @property
    def has_invalid(self) -> bool:
        """Whether code ``0`` is a reserved "no measurement" sentinel (valid depth then uses ``[1, max_raw]``)."""
        return self._has_invalid

    @property
    def depth_type(self) -> DepthType:
        """``"z_depth"`` (planar) or ``"ray_distance"`` (euclidean): what the stored scalar measures."""
        return self._depth_type

    @property
    def max_raw(self) -> int:
        """The largest storable integer, ``2 ** depth_bits - 1`` (i.e. ``255`` or ``65535``)."""
        return (1 << self._depth_bits) - 1

    @property
    def _code_offset(self) -> int:
        """The smallest valid code: ``1`` when ``has_invalid`` reserves ``0``, else ``0``."""
        return 1 if self._has_invalid else 0

    @property
    def _code_span(self) -> int:
        """The number of quantization steps across the valid code range ``[offset, max_raw]``."""
        return self.max_raw - self._code_offset

    @property
    def depth_dtype(self) -> np.dtype:
        """The numpy dtype of the stored raster: ``uint8`` or ``uint16``."""
        return _DEPTH_DTYPES[self._depth_bits]

    @property
    def depth_resolution(self) -> float:
        """Metres per integer unit for the ``linear`` transform, ``max_depth / code_span``.

        For ``inverse`` the step is not constant (fine near, coarse far); this returns the same nominal
        value only as a rough far-plane-ish scale, and the half-step error bound does not apply.
        """
        return self._max_depth / self._code_span

    @property
    def channel_type(self) -> CameraChannelType:
        """Always ``CameraChannelType.DEPTH``."""
        return CameraChannelType.DEPTH

    # ------------------------------------------------------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------------------------------------------------------

    def _depth_to_unit(self, depth_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map clipped metric depth in ``[min_depth, max_depth]`` onto ``[0, 1]``, monotonically increasing.

        ``linear`` is anchored at absolute zero (``depth / max_depth``); ``inverse`` spaces the unit
        interval uniformly in disparity so ``t = 0`` at ``min_depth`` and ``t = 1`` at ``max_depth``.
        """
        if self._depth_transform == "linear":
            return depth_m / self._max_depth
        inv_near = 1.0 / self._min_depth
        inv_far = 1.0 / self._max_depth
        return (inv_near - 1.0 / depth_m) / (inv_near - inv_far)

    def _unit_to_depth(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inverse of :meth:`_depth_to_unit`: map ``[0, 1]`` back onto metric depth."""
        if self._depth_transform == "linear":
            return t * self._max_depth
        inv_near = 1.0 / self._min_depth
        inv_far = 1.0 / self._max_depth
        return 1.0 / (inv_near - t * (inv_near - inv_far))

    def encode_depth(self, depth: npt.NDArray[np.floating]) -> npt.NDArray[np.unsignedinteger]:
        """Quantize a metric depth map ``(H, W)`` in metres into the stored integer raster.

        Clips to ``[min_depth, max_depth]``, maps onto ``[0, 1]`` via ``depth_transform``, rescales onto
        the valid codes ``[offset, max_raw]``, and rounds to nearest (rather than truncating, which
        would bias every pixel downward by half a step on average).

        With ``has_invalid=False`` (default), non-finite values (``NaN``/``inf``, e.g. an unrendered
        pixel) clamp to the near/far plane. With ``has_invalid=True``, any non-finite or non-positive
        pixel is instead written as the ``0`` sentinel.

        :param depth: A 2D ``(H, W)`` float array of depths in metres.
        :return: A 2D ``(H, W)`` array of ``depth_dtype``.
        """
        assert depth.ndim == 2, f"Depth map must be a single-channel (H, W) array, got shape {depth.shape}."
        depth = np.asarray(depth, dtype=np.float64)
        # A pixel is "no measurement" iff it lacks a real, finite, positive depth. Captured before the
        # clamp below folds NaN/inf/negatives into the valid range.
        invalid = ~(np.isfinite(depth) & (depth > 0.0)) if self._has_invalid else None
        # nan -> max_depth (an unrendered / infinitely distant pixel is "as far as we can say"), and
        # +-inf -> the clip bounds. Done before the clip so NaN doesn't propagate through it.
        depth_m = np.nan_to_num(depth, nan=self._max_depth, posinf=self._max_depth, neginf=self._min_depth)
        depth_m = np.clip(depth_m, self._min_depth, self._max_depth)
        raw = self._code_offset + np.round(self._depth_to_unit(depth_m) * self._code_span)
        if invalid is not None:
            raw[invalid] = 0
        return raw.astype(self.depth_dtype)

    def decode_depth(self, raw: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
        """Dequantize a stored integer raster back to metric depth in metres.

        The inverse of ``encode_depth``, up to the quantization error. Pixels that saturated on encode
        decode to exactly ``max_depth``, not to their true distance. When ``has_invalid`` is set, the
        ``0`` sentinel decodes to ``NaN``.

        :param raw: A 2D ``(H, W)`` integer array as returned by ``encode_depth``.
        :return: A 2D ``(H, W)`` float32 array of depths in metres (``NaN`` for invalid pixels).
        """
        t = (np.asarray(raw, dtype=np.float64) - self._code_offset) / self._code_span
        depth = self._unit_to_depth(t).astype(np.float32)
        if self._has_invalid:
            depth = np.where(np.asarray(raw) == 0, np.float32(np.nan), depth)
        return depth

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
            "depth_transform": self._depth_transform,
            "min_depth": self._min_depth,
            "has_invalid": self._has_invalid,
            "depth_type": self._depth_type,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> DepthCameraMetadata:
        """Construct a ``DepthCameraMetadata`` from a dictionary.

        The transform/sentinel/type keys default to the historic behaviour (linear, no near clip, no
        invalid sentinel, z-depth) so ``.arrow`` files written before these knobs existed still read.
        """
        return cls(
            camera_metadata=camera_metadata_from_dict(data_dict["camera_metadata"]),
            max_depth=data_dict["max_depth"],
            depth_bits=data_dict["depth_bits"],
            depth_transform=data_dict.get("depth_transform", "linear"),
            min_depth=data_dict.get("min_depth", 0.0),
            has_invalid=data_dict.get("has_invalid", False),
            depth_type=data_dict.get("depth_type", "z_depth"),
        )

    def __repr__(self) -> str:
        return (
            f"DepthCameraMetadata(camera_id={self.camera_id}, max_depth={self._max_depth}, "
            f"min_depth={self._min_depth}, depth_bits={self._depth_bits}, "
            f"depth_transform={self._depth_transform!r}, has_invalid={self._has_invalid}, "
            f"depth_type={self._depth_type!r})"
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
