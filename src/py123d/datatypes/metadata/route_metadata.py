from __future__ import annotations

from typing import Any, Dict

import py123d
from py123d.datatypes.metadata.base_metadata import BaseMetadata


class RouteMetadata(BaseMetadata):
    """Metadata for a log's driven-route polyline (``route.arrow``).

    The route polyline is the ego's driven path over the whole log, resampled at a
    fixed arc-length resolution. It is derived data computed by the log writer from
    ego odometry, not a sensor modality.
    """

    __slots__ = ("_resolution_m", "_total_arc_m", "_source", "_version")

    def __init__(
        self,
        resolution_m: float,
        total_arc_m: float,
        source: str = "ego_state_se3",
        version: str = str(py123d.__version__),
    ):
        """Initialize a RouteMetadata instance.

        :param resolution_m: Arc-length spacing between consecutive polyline vertices in meters.
        :param total_arc_m: Total arc-length of the driven route in meters.
        :param source: Modality key the route was derived from, defaults to ``"ego_state_se3"``.
        :param version: Version of the py123d library used to create this route metadata,
            defaults to str(py123d.__version__)
        """
        self._resolution_m = resolution_m
        self._total_arc_m = total_arc_m
        self._source = source
        self._version = version

    @property
    def resolution_m(self) -> float:
        """Arc-length spacing between consecutive polyline vertices in meters."""
        return self._resolution_m

    @property
    def total_arc_m(self) -> float:
        """Total arc-length of the driven route in meters."""
        return self._total_arc_m

    @property
    def source(self) -> str:
        """Modality key the route was derived from."""
        return self._source

    @property
    def version(self) -> str:
        """Version of the py123d library used to create this route metadata."""
        return self._version

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> RouteMetadata:
        """Create a RouteMetadata instance from a dictionary.

        :param data_dict: A dictionary representation of a RouteMetadata instance.
        :return: A RouteMetadata instance.
        """
        return RouteMetadata(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RouteMetadata instance to a dictionary.

        :return: A dictionary representation of the RouteMetadata instance.
        """
        return {slot.lstrip("_"): getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"RouteMetadata("
            f"resolution_m={self.resolution_m}, "
            f"total_arc_m={self.total_arc_m}, "
            f"source={self.source!r}, "
            f"version={self.version!r}"
            f")"
        )
