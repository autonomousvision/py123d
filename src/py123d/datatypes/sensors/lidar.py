from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.conversion.registry import LIDAR_INDEX_REGISTRY, LiDARIndex
from py123d.geometry import PoseSE3


class LiDARType(SerialIntEnum):
    """Enumeration of LiDAR sensors, in multi-sensor setups."""

    LIDAR_UNKNOWN = 0
    """Unknown LiDAR type."""

    LIDAR_MERGED = 1
    """Merged LiDAR type."""

    LIDAR_TOP = 2
    """Top-facing LiDAR type."""

    LIDAR_FRONT = 3
    """Front-facing LiDAR type."""

    LIDAR_SIDE_LEFT = 4
    """Left-side LiDAR type."""

    LIDAR_SIDE_RIGHT = 5
    """Right-side LiDAR type."""

    LIDAR_BACK = 6
    """Back-facing LiDAR type."""

    LIDAR_DOWN = 7
    """Down-facing LiDAR type."""


class LiDARMetadata:
    """Metadata for LiDAR sensor, static for a given sensor."""

    __slots__ = ("_lidar_type", "_lidar_index", "_extrinsic")

    def __init__(
        self,
        lidar_type: LiDARType,
        lidar_index: Type[LiDARIndex],
        extrinsic: Optional[PoseSE3] = None,
    ):
        """Initialize LiDAR metadata.

        :param lidar_type: The type of the LiDAR sensor.
        :param lidar_index: The indexing schema of the LiDAR point cloud.
        :param extrinsic: The extrinsic pose of the LiDAR sensor, defaults to None
        """
        self._lidar_type = lidar_type
        self._lidar_index = lidar_index
        self._extrinsic = extrinsic

    @property
    def lidar_type(self) -> LiDARType:
        """The type of the LiDAR sensor."""
        return self._lidar_type

    @property
    def lidar_index(self) -> LiDARIndex:
        """The indexing schema of the LiDAR point cloud."""
        return self._lidar_index

    @property
    def extrinsic(self) -> Optional[PoseSE3]:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the LiDAR sensor, relative to the vehicle frame."""
        return self._extrinsic

    @classmethod
    def from_dict(cls, data_dict: dict) -> LiDARMetadata:
        """Construct the LiDAR metadata from a dictionary.

        :param data_dict: A dictionary containing LiDAR metadata.
        :raises ValueError: If the dictionary is missing required fields or contains invalid data.
        :return: An instance of LiDARMetadata.
        """
        lidar_type = LiDARType[data_dict["lidar_type"]]
        if data_dict["lidar_index"] not in LIDAR_INDEX_REGISTRY:
            raise ValueError(f"Unknown lidar index: {data_dict['lidar_index']}")
        lidar_index_class = LIDAR_INDEX_REGISTRY[data_dict["lidar_index"]]
        extrinsic = PoseSE3.from_list(data_dict["extrinsic"]) if data_dict["extrinsic"] is not None else None
        return cls(lidar_type=lidar_type, lidar_index=lidar_index_class, extrinsic=extrinsic)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the LiDAR metadata to a dictionary.

        :return: A dictionary representation of the LiDAR metadata.
        """
        return {
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index.__name__,
            "extrinsic": self.extrinsic.tolist() if self.extrinsic is not None else None,
        }


class LiDAR:
    """Data structure for LiDAR point cloud data and associated metadata."""

    __slots__ = ("_metadata", "_point_cloud")

    def __init__(self, metadata: LiDARMetadata, point_cloud: npt.NDArray[np.float32]) -> None:
        """Initialize LiDAR data structure.

        :param metadata: LiDAR metadata.
        :param point_cloud: LiDAR point cloud as an NxM numpy array, where N is the number of points
            and M is the number of attributes per point as defined by the :class:`~py123d.conversion.registry.LiDARIndex`.
        """
        self._metadata = metadata
        self._point_cloud = point_cloud

    @property
    def metadata(self) -> LiDARMetadata:
        """The :class:`LiDARMetadata` associated with this LiDAR recording."""
        return self._metadata

    @property
    def point_cloud(self) -> npt.NDArray[np.float32]:
        """The raw point cloud as an NxM numpy array,
        where N is the number of points and M is the number of attributes per point,
        as defined by the :class:`~py123d.conversion.registry.LiDARIndex`. Point cloud in vehicle frame.
        """
        return self._point_cloud

    @property
    def xyz(self) -> npt.NDArray[np.float32]:
        """The point cloud as an Nx3 array of x, y, z coordinates."""
        return self._point_cloud[:, self.metadata.lidar_index.XYZ]

    @property
    def xy(self) -> npt.NDArray[np.float32]:
        """The point cloud as an Nx2 array of x, y coordinates."""
        return self._point_cloud[:, self.metadata.lidar_index.XY]

    @property
    def intensity(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of intensity values, if available."""
        intensity: Optional[npt.NDArray[np.float32]] = None
        if hasattr(self._metadata.lidar_index, "INTENSITY"):
            intensity = self._point_cloud[:, self._metadata.lidar_index.INTENSITY]
        return intensity

    @property
    def range(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of range values, if available."""
        range: Optional[npt.NDArray[np.float32]] = None
        if hasattr(self._metadata.lidar_index, "RANGE"):
            range = self._point_cloud[:, self._metadata.lidar_index.RANGE]
        return range

    @property
    def elongation(self) -> Optional[npt.NDArray[np.float32]]:
        """The point cloud as an Nx1 array of elongation values, if available."""
        elongation: Optional[npt.NDArray[np.float32]] = None
        if hasattr(self._metadata.lidar_index, "ELONGATION"):
            elongation = self._point_cloud[:, self._metadata.lidar_index.ELONGATION]
        return elongation

    @property
    def ring(self) -> Optional[npt.NDArray[np.int32]]:
        """The point cloud as an Nx1 array of ring values, if available."""
        ring: Optional[npt.NDArray[np.int32]] = None
        if hasattr(self._metadata.lidar_index, "RING"):
            ring = self._point_cloud[:, self._metadata.lidar_index.RING]
        return ring
