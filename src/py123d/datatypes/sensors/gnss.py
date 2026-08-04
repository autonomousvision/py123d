from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3


class GnssMetadata(BaseModalityMetadata):
    """Metadata for a GNSS receiver, static for a given sensor.

    Mirrors the shape of ``sensor_msgs/msg/NavSatFix``. The optional :attr:`datum_lla` records
    the geodetic reference point that the log's local metric frame was anchored to (typically
    the first valid fix), so local x/y/z coordinates in the log can be mapped back to earth
    coordinates.
    """

    __slots__ = ("_gnss_name", "_gnss_id", "_gnss_to_imu_se3", "_datum_lla")

    def __init__(
        self,
        gnss_name: str,
        gnss_id: Optional[str] = None,
        gnss_to_imu_se3: PoseSE3 = PoseSE3.identity(),
        datum_lla: Optional[Tuple[float, float, float]] = None,
    ):
        """Initialize GNSS metadata.

        :param gnss_name: The name of the GNSS receiver from the dataset.
        :param gnss_id: Optional identifier to distinguish multiple receivers in one rig. None
            (the default) means the log has a single receiver and the modality key is ``gnss``.
        :param gnss_to_imu_se3: The extrinsic pose of the GNSS antenna relative to the IMU frame.
        :param datum_lla: Optional (latitude_deg, longitude_deg, altitude_m) reference point of
            the log's local metric frame.
        """
        self._gnss_name = gnss_name
        self._gnss_id = gnss_id
        self._gnss_to_imu_se3 = gnss_to_imu_se3
        self._datum_lla = datum_lla

    @property
    def gnss_name(self) -> str:
        """The name of the GNSS receiver from the dataset."""
        return self._gnss_name

    @property
    def gnss_id(self) -> Optional[str]:
        """Optional identifier to distinguish multiple GNSS receivers in one rig."""
        return self._gnss_id

    @property
    def gnss_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the GNSS antenna, relative to the IMU frame."""
        return self._gnss_to_imu_se3

    @property
    def datum_lla(self) -> Optional[Tuple[float, float, float]]:
        """The (latitude_deg, longitude_deg, altitude_m) datum of the log's local frame, if recorded."""
        return self._datum_lla

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.GNSS

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._gnss_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> GnssMetadata:
        """Construct the GNSS metadata from a dictionary.

        :param data_dict: A dictionary containing GNSS metadata.
        :return: An instance of GnssMetadata.
        """
        datum = data_dict.get("datum_lla")
        return GnssMetadata(
            gnss_name=data_dict["gnss_name"],
            gnss_id=data_dict.get("gnss_id"),
            gnss_to_imu_se3=PoseSE3.from_list(data_dict["gnss_to_imu_se3"]),
            datum_lla=tuple(datum) if datum is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GNSS metadata to a dictionary.

        :return: A dictionary representation of the GNSS metadata.
        """
        return {
            "gnss_name": self._gnss_name,
            "gnss_id": self._gnss_id,
            "gnss_to_imu_se3": self._gnss_to_imu_se3.tolist(),
            "datum_lla": list(self._datum_lla) if self._datum_lla is not None else None,
        }


class Gnss(BaseModality):
    """Data structure for a single GNSS fix and associated metadata.

    Field semantics follow ``sensor_msgs/msg/NavSatFix``: WGS-84 geodetic coordinates, ENU
    position covariance in m^2, and the NavSatStatus fix status / service constants
    (status: -1 no fix, 0 fix, 1 SBAS fix, 2 GBAS fix).
    """

    __slots__ = (
        "_timestamp",
        "_metadata",
        "_latitude",
        "_longitude",
        "_altitude",
        "_position_covariance",
        "_position_covariance_type",
        "_status",
        "_service",
    )

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: GnssMetadata,
        latitude: float,
        longitude: float,
        altitude: float,
        position_covariance: Optional[npt.NDArray[np.float64]] = None,
        position_covariance_type: Optional[int] = None,
        status: Optional[int] = None,
        service: Optional[int] = None,
    ) -> None:
        """Initialize a GNSS fix.

        :param timestamp: The timestamp of the fix.
        :param metadata: The GNSS metadata.
        :param latitude: WGS-84 latitude in degrees.
        :param longitude: WGS-84 longitude in degrees.
        :param altitude: Altitude in meters above the WGS-84 ellipsoid.
        :param position_covariance: Optional row-major 3x3 ENU position covariance in m^2, flattened to (9,).
        :param position_covariance_type: Optional NavSatFix covariance type constant.
        :param status: Optional NavSatStatus fix status constant.
        :param service: Optional NavSatStatus service bitmask.
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._latitude = latitude
        self._longitude = longitude
        self._altitude = altitude
        self._position_covariance = position_covariance
        self._position_covariance_type = position_covariance_type
        self._status = status
        self._service = service

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this GNSS fix."""
        return self._timestamp

    @property
    def metadata(self) -> GnssMetadata:
        """The :class:`GnssMetadata` associated with this GNSS fix."""
        return self._metadata

    @property
    def latitude(self) -> float:
        """WGS-84 latitude in degrees."""
        return self._latitude

    @property
    def longitude(self) -> float:
        """WGS-84 longitude in degrees."""
        return self._longitude

    @property
    def altitude(self) -> float:
        """Altitude in meters above the WGS-84 ellipsoid."""
        return self._altitude

    @property
    def lla(self) -> npt.NDArray[np.float64]:
        """The fix as a (3,) array of (latitude_deg, longitude_deg, altitude_m)."""
        return np.array([self._latitude, self._longitude, self._altitude], dtype=np.float64)

    @property
    def position_covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Row-major 3x3 ENU position covariance in m^2 flattened to (9,), or None if not provided."""
        return self._position_covariance

    @property
    def position_covariance_type(self) -> Optional[int]:
        """NavSatFix covariance type constant, or None if not provided."""
        return self._position_covariance_type

    @property
    def status(self) -> Optional[int]:
        """NavSatStatus fix status constant (-1 no fix, 0 fix, 1 SBAS, 2 GBAS), or None if not provided."""
        return self._status

    @property
    def service(self) -> Optional[int]:
        """NavSatStatus service bitmask, or None if not provided."""
        return self._service
