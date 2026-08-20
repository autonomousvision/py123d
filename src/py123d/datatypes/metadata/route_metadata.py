from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import py123d
from py123d.datatypes.metadata.base_metadata import BaseMetadata


@dataclass(frozen=True)
class RouteMetadata(BaseMetadata):
    """Metadata for a log's route polyline (``route.arrow``): the driven path derived from
    ego odometry or an explicitly provided route, resampled at a fixed arc-length resolution.

    :param resolution_m: Arc-length spacing between consecutive polyline vertices in meters.
    :param total_arc_m: Total arc-length of the route in meters.
    :param source: Origin of the route: the modality key it was derived from, or ``"provided"``.
    :param version: py123d version that wrote the route.
    """

    resolution_m: float
    total_arc_m: float
    source: str = "ego_state_se3"
    version: str = field(default_factory=lambda: str(py123d.__version__))

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> RouteMetadata:
        """Inherited, see superclass."""
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return asdict(self)
