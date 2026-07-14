from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PointCloudCodecConfig:
    """Encoder settings for the point-cloud codecs, shared by the Lidar and Radar writers.

    The defaults reproduce the historical hard-coded encoder behaviour, so logs written without an
    explicit config are byte-for-byte unchanged.
    """

    # --- LAZ ---
    # LAS point format. Format 0 carries only xyz + the base attributes; formats 1/3 additionally
    # reserve room for GPS time and RGB, which cost bytes even when left unset.
    laz_point_format: int = 3
    # Quantization step of the LAS integer grid, in meters, per axis. This is the dominant driver of
    # the encoded size: LASzip stores xyz as int32 multiples of the scale, so a coarser scale
    # directly shortens the arithmetic-coded residuals.
    laz_scales: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    # Origin of the LAS integer grid, in meters, per axis.
    laz_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- Draco ---
    draco_quantization_bits: int = 16
    # Range: 0 (fastest) to 10 (slowest, best compression).
    draco_compression_level: int = 7
    # -1 uses Draco's default range.
    draco_quantization_range: int = -1
    draco_preserve_order: bool = True

    # --- Arrow IPC ---
    # Codec-specific compression level; None uses the pyarrow default for the codec.
    ipc_compression_level: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.laz_point_format in {0, 1, 2, 3, 6, 7, 8}, (
            f"Unsupported LAS point format, got {self.laz_point_format}."
        )
        assert len(self.laz_scales) == 3 and all(scale > 0.0 for scale in self.laz_scales), (
            f"LAZ scales must be three positive values, got {self.laz_scales}."
        )
        assert len(self.laz_offsets) == 3, f"LAZ offsets must be three values, got {self.laz_offsets}."
        assert 1 <= self.draco_quantization_bits <= 30, (
            f"Draco quantization bits must be in [1, 30], got {self.draco_quantization_bits}."
        )
        assert 0 <= self.draco_compression_level <= 10, (
            f"Draco compression level must be in [0, 10], got {self.draco_compression_level}."
        )
