from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from py123d.datatypes.metadata.computed_from import ComputedFrom
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
    :param computed_from: Record of the producer and of every log input the route was
        computed from. The route is a cache of that computation, so the record is required:
        without it a reader cannot tell that the ego odometry has changed underneath it.
    :param source: Origin of the route: the modality key it was derived from, or ``"provided"``.
    """

    resolution_m: float
    total_arc_m: float
    polyline_x: List[float]
    polyline_y: List[float]
    polyline_z: List[float]
    computed_from: ComputedFrom
    source: str = "ego_state_se3"

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
        if "computed_from" not in data_dict:
            raise ValueError(
                "Route metadata without a 'computed_from' record. The route was written by code that predates "
                "input tracking; recompute it so its inputs can be verified."
            )
        return cls(
            resolution_m=data_dict["resolution_m"],
            total_arc_m=data_dict["total_arc_m"],
            polyline_x=data_dict["polyline_x"],
            polyline_y=data_dict["polyline_y"],
            polyline_z=data_dict["polyline_z"],
            computed_from=ComputedFrom.from_dict(data_dict["computed_from"]),
            source=data_dict["source"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {
            "resolution_m": self.resolution_m,
            "total_arc_m": self.total_arc_m,
            "polyline_x": self.polyline_x,
            "polyline_y": self.polyline_y,
            "polyline_z": self.polyline_z,
            "computed_from": self.computed_from.to_dict(),
            "source": self.source,
        }
