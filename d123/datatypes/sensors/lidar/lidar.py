from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import numpy.typing as npt

from d123.common.utils.enums import SerialIntEnum
from d123.conversion.utils.sensor_utils.lidar_index_registry import LIDAR_INDEX_REGISTRY, LiDARIndex
from d123.geometry import StateSE3


class LiDARType(SerialIntEnum):

    LIDAR_UNKNOWN = 0
    LIDAR_MERGED = 1
    LIDAR_TOP = 2
    LIDAR_FRONT = 3
    LIDAR_SIDE_LEFT = 4
    LIDAR_SIDE_RIGHT = 5
    LIDAR_BACK = 6


@dataclass
class LiDARMetadata:

    lidar_type: LiDARType
    lidar_index: Type[LiDARIndex]
    extrinsic: Optional[StateSE3] = None
    # TODO: add identifier if point cloud is returned in lidar or ego frame.

    def to_dict(self) -> dict:
        return {
            "lidar_type": self.lidar_type.name,
            "lidar_index": self.lidar_index.__name__,
            "extrinsic": self.extrinsic.tolist() if self.extrinsic is not None else None,
        }

    @classmethod
    def from_dict(cls, data_dict: dict) -> LiDARMetadata:
        lidar_type = LiDARType[data_dict["lidar_type"]]
        if data_dict["lidar_index"] not in LIDAR_INDEX_REGISTRY:
            raise ValueError(f"Unknown lidar index: {data_dict['lidar_index']}")
        lidar_index_class = LIDAR_INDEX_REGISTRY[data_dict["lidar_index"]]
        extrinsic = StateSE3.from_list(data_dict["extrinsic"]) if data_dict["extrinsic"] is not None else None
        return cls(lidar_type=lidar_type, lidar_index=lidar_index_class, extrinsic=extrinsic)


@dataclass
class LiDAR:

    metadata: LiDARMetadata
    point_cloud: npt.NDArray[np.float32]

    @property
    def xyz(self) -> npt.NDArray[np.float32]:
        """
        Returns the point cloud as an Nx3 array of x, y, z coordinates.
        """
        return self.point_cloud[self.metadata.lidar_index.XYZ].T

    @property
    def xy(self) -> npt.NDArray[np.float32]:
        """
        Returns the point cloud as an Nx2 array of x, y coordinates.
        """
        return self.point_cloud[self.metadata.lidar_index.XY].T

    @property
    def intensity(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Returns the intensity values of the LiDAR point cloud if available.
        Returns None if intensity is not part of the point cloud.
        """
        if hasattr(self.metadata.lidar_index, "INTENSITY"):
            return self.point_cloud[self.metadata.lidar_index.INTENSITY]
        return None

    @property
    def range(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Returns the range values of the LiDAR point cloud if available.
        Returns None if range is not part of the point cloud.
        """
        if hasattr(self.metadata.lidar_index, "RANGE"):
            return self.point_cloud[self.metadata.lidar_index.RANGE]
        return None

    @property
    def elongation(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Returns the elongation values of the LiDAR point cloud if available.
        Returns None if elongation is not part of the point cloud.
        """
        if hasattr(self.metadata.lidar_index, "ELONGATION"):
            return self.point_cloud[self.metadata.lidar_index.ELONGATION]
        return None
