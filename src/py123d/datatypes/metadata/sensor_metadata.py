"""Backward-compat shim — these types are deprecated and will be removed.

Use Dict[CameraID, CameraMetadata] / Dict[LidarID, LidarMetadata] instead.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict

from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata


class PinholeCameraMetadatas(Mapping[PinholeCameraID, PinholeCameraMetadata]):
    __slots__ = ("_data",)

    def __init__(self, pinhole_camera_metadata_dict: Dict[PinholeCameraID, PinholeCameraMetadata]):
        self._data = pinhole_camera_metadata_dict

    def __getitem__(self, key: PinholeCameraID) -> PinholeCameraMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[PinholeCameraID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> PinholeCameraMetadatas:
        return PinholeCameraMetadatas(
            pinhole_camera_metadata_dict={
                PinholeCameraID(int(k)): PinholeCameraMetadata.from_dict(v) for k, v in data_dict.items()
            }
        )


class FisheyeMEICameraMetadatas(Mapping[FisheyeMEICameraID, FisheyeMEICameraMetadata]):
    __slots__ = ("_data",)

    def __init__(self, fisheye_mei_camera_metadata_dict: Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]):
        self._data = fisheye_mei_camera_metadata_dict

    def __getitem__(self, key: FisheyeMEICameraID) -> FisheyeMEICameraMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[FisheyeMEICameraID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> FisheyeMEICameraMetadatas:
        return FisheyeMEICameraMetadatas(
            fisheye_mei_camera_metadata_dict={
                FisheyeMEICameraID(int(k)): FisheyeMEICameraMetadata.from_dict(v) for k, v in data_dict.items()
            }
        )


class LidarMetadatas(Mapping[LidarID, LidarMetadata]):
    __slots__ = ("_data",)

    def __init__(self, lidar_metadata_dict: Dict[LidarID, LidarMetadata]):
        self._data = lidar_metadata_dict

    def __getitem__(self, key: LidarID) -> LidarMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[LidarID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> LidarMetadatas:
        return LidarMetadatas(
            lidar_metadata_dict={LidarID(int(k)): LidarMetadata.from_dict(v) for k, v in data_dict.items()}
        )
