from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

import py123d
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata, ModalityType


@dataclass(frozen=True)
class RouteMetadata(BaseModalityMetadata):
    """Metadata of the ``route_position`` modality: the static route polyline plus its
    parameterization, while the modality rows carry the dynamic per-frame positions.

    The route is the driven path derived from ego odometry or an explicitly provided
    route, resampled at a fixed arc-length resolution. Vertices are stored as plain
    float lists (msgpack-encodable); the arc-length of vertex ``i`` is implicit:
    ``i * resolution_m``, with the final vertex exactly at ``total_arc_m``.

    :param resolution_m: Arc-length spacing between consecutive polyline vertices in meters.
    :param total_arc_m: Total arc-length of the route in meters.
    :param polyline_x: X coordinate per polyline vertex, in the ego odometry frame.
    :param polyline_y: Y coordinate per polyline vertex.
    :param polyline_z: Z coordinate per polyline vertex.
    :param source: Origin of the route: the modality key it was derived from, or ``"provided"``.
    :param version: py123d version that wrote the route.
    """

    resolution_m: float
    total_arc_m: float
    polyline_x: List[float]
    polyline_y: List[float]
    polyline_z: List[float]
    source: str = "ego_state_se3"
    version: str = field(default_factory=lambda: str(py123d.__version__))

    @property
    def modality_type(self) -> ModalityType:
        """Inherited, see superclass."""
        return ModalityType.ROUTE_POSITION

    @cached_property
    def polyline_xyz(self) -> npt.NDArray[np.float64]:
        """The polyline vertices as an array of shape (K, 3)."""
        return np.stack(
            [
                np.asarray(self.polyline_x, dtype=np.float64),
                np.asarray(self.polyline_y, dtype=np.float64),
                np.asarray(self.polyline_z, dtype=np.float64),
            ],
            axis=1,
        )

    @cached_property
    def polyline_arc_m(self) -> npt.NDArray[np.float64]:
        """Arc-length per polyline vertex, shape (K,): uniform steps of ``resolution_m``
        with the final vertex exactly at ``total_arc_m``."""
        arc = np.arange(len(self.polyline_x), dtype=np.float64) * self.resolution_m
        if len(arc) > 0:
            arc[-1] = self.total_arc_m
        return arc

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> RouteMetadata:
        """Inherited, see superclass."""
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return asdict(self)
