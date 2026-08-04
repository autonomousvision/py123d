from __future__ import annotations

from typing import Any, Dict, Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3


class BarometerMetadata(BaseModalityMetadata):
    """Metadata for a barometric pressure sensor, static for a given sensor."""

    __slots__ = ("_barometer_name", "_barometer_id", "_barometer_to_imu_se3")

    def __init__(
        self,
        barometer_name: str,
        barometer_id: Optional[str] = None,
        barometer_to_imu_se3: PoseSE3 = PoseSE3.identity(),
    ):
        """Initialize barometer metadata.

        :param barometer_name: The name of the barometer sensor from the dataset.
        :param barometer_id: Optional identifier to distinguish multiple barometers in one rig.
            None (the default) means the log has a single barometer and the modality key is
            ``barometer``.
        :param barometer_to_imu_se3: The extrinsic pose of the sensor relative to the IMU frame.
        """
        self._barometer_name = barometer_name
        self._barometer_id = barometer_id
        self._barometer_to_imu_se3 = barometer_to_imu_se3

    @property
    def barometer_name(self) -> str:
        """The name of the barometer sensor from the dataset."""
        return self._barometer_name

    @property
    def barometer_id(self) -> Optional[str]:
        """Optional identifier to distinguish multiple barometers in one rig."""
        return self._barometer_id

    @property
    def barometer_to_imu_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the sensor, relative to the IMU frame."""
        return self._barometer_to_imu_se3

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.BAROMETER

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        return self._barometer_id

    @classmethod
    def from_dict(cls, data_dict: dict) -> BarometerMetadata:
        """Construct the barometer metadata from a dictionary.

        :param data_dict: A dictionary containing barometer metadata.
        :return: An instance of BarometerMetadata.
        """
        return BarometerMetadata(
            barometer_name=data_dict["barometer_name"],
            barometer_id=data_dict.get("barometer_id"),
            barometer_to_imu_se3=PoseSE3.from_list(data_dict["barometer_to_imu_se3"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the barometer metadata to a dictionary.

        :return: A dictionary representation of the barometer metadata.
        """
        return {
            "barometer_name": self._barometer_name,
            "barometer_id": self._barometer_id,
            "barometer_to_imu_se3": self._barometer_to_imu_se3.tolist(),
        }


class Barometer(BaseModality):
    """Data structure for a single barometric pressure measurement and associated metadata.

    Pressure is in Pascal, following ``sensor_msgs/msg/FluidPressure``. The derived
    mean-sea-level altitude and the environmental readings are optional.
    """

    __slots__ = ("_timestamp", "_metadata", "_pressure", "_msl_altitude", "_temperature", "_humidity")

    def __init__(
        self,
        timestamp: Timestamp,
        metadata: BarometerMetadata,
        pressure: float,
        msl_altitude: Optional[float] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
    ) -> None:
        """Initialize a barometer measurement.

        :param timestamp: The timestamp of the measurement.
        :param metadata: The barometer metadata.
        :param pressure: Barometric pressure in Pascal.
        :param msl_altitude: Optional mean-sea-level altitude derived from pressure, in meters.
        :param temperature: Optional sensor temperature in degrees Celsius.
        :param humidity: Optional relative humidity in percent (0-100).
        """
        self._timestamp = timestamp
        self._metadata = metadata
        self._pressure = pressure
        self._msl_altitude = msl_altitude
        self._temperature = temperature
        self._humidity = humidity

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp associated with this barometer measurement."""
        return self._timestamp

    @property
    def metadata(self) -> BarometerMetadata:
        """The :class:`BarometerMetadata` associated with this barometer measurement."""
        return self._metadata

    @property
    def pressure(self) -> float:
        """Barometric pressure in Pascal."""
        return self._pressure

    @property
    def msl_altitude(self) -> Optional[float]:
        """Mean-sea-level altitude derived from pressure in meters, or None if not provided."""
        return self._msl_altitude

    @property
    def temperature(self) -> Optional[float]:
        """Sensor temperature in degrees Celsius, or None if not provided."""
        return self._temperature

    @property
    def humidity(self) -> Optional[float]:
        """Relative humidity in percent (0-100), or None if not provided."""
        return self._humidity
