from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from py123d.geometry.geometry_index import Point2DIndex


class OccupancyMap2D:
    """Class to represent a 2D occupancy map of shapely geometries using an str-tree for efficient spatial queries."""

    def __init__(
        self,
        geometries: Sequence[BaseGeometry],
        ids: Optional[Union[Sequence[str], Sequence[int]]] = None,
        node_capacity: int = 10,
    ):
        """Constructs a 2D occupancy map of shapely geometries using an str-tree for efficient spatial queries.

        :param geometries: list/array of shapely geometries
        :param ids: optional list of geometry identifiers, either strings or integers
        :param node_capacity: max number of child nodes in str-tree, defaults to 10
        """

        assert ids is None or len(ids) == len(geometries), "Length of ids must match length of geometries"
        if ids is not None:
            assert all(isinstance(id, (str, int)) for id in ids), "IDs must be either strings or integers"

        self._ids: Sequence[Union[str, int]] = ids if ids is not None else [str(idx) for idx in range(len(geometries))]
        self._id_to_idx: Dict[Union[str, int], int] = {id: idx for idx, id in enumerate(self._ids)}

        self._geometries = geometries
        self._node_capacity = node_capacity
        self._str_tree = STRtree(self._geometries, node_capacity)

    @classmethod
    def from_dict(
        cls,
        geometry_dict: Union[Dict[str, BaseGeometry], Dict[int, BaseGeometry]],
        node_capacity: int = 10,
    ) -> OccupancyMap2D:
        """Constructs a 2D occupancy map from a dictionary of geometries.

        :param geometry_dict: Dictionary mapping geometry identifiers to shapely geometries
        :param node_capacity: Max number of child nodes in str-tree, defaults to 10
        :return: OccupancyMap2D instance
        """
        ids = list(geometry_dict.keys())
        geometries = list(geometry_dict.values())
        return cls(geometries=geometries, ids=ids, node_capacity=node_capacity)  # type: ignore

    def __getitem__(self, id: Union[str, int]) -> BaseGeometry:
        """Retrieves geometry given an ID.

        :param id: geometry identifier
        :return: Geometry of ID.
        """
        return self._geometries[self._id_to_idx[id]]

    def __len__(self) -> int:
        """
        :return: Number of geometries in the occupancy map.
        """
        return len(self._ids)

    @property
    def ids(self) -> Union[List[str], List[int]]:
        """Getter for geometry IDs in occupancy map

        :return: list of IDs
        """
        return self._ids  # type: ignore

    @property
    def geometries(self) -> Sequence[BaseGeometry]:
        """Getter for geometries in occupancy map.

        :return: list of geometries
        """
        return self._geometries

    @property
    def id_to_idx(self) -> Dict[Union[int, str], int]:
        """Mapping from geometry IDs to indices in the occupancy map.

        :return: dictionary of IDs and indices
        """
        return self._id_to_idx

    def intersects(self, geometry: BaseGeometry) -> Union[List[str], List[int]]:
        """Searches for intersecting geometries in the occupancy map.

        :param geometry: geometries to query
        :return: list of IDs for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._ids[idx] for idx in indices]

    def query(
        self,
        geometry: Union[BaseGeometry, np.ndarray],
        predicate: Optional[
            Literal[
                "intersects",
                "within",
                "dwithin",
                "contains",
                "overlaps",
                "crosses",
                "touches",
                "covers",
                "covered_by",
            ]
        ] = None,
        distance: Optional[float] = None,
    ) -> npt.NDArray[np.int64]:
        """Queries the str-tree for geometries that match the given predicate with the input geometry.

        :param geometry: Geometry or array_like
        :param predicate: {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches', 'covers', \
            'covered_by', 'contains_properly', 'dwithin'}, defaults to None
        :param distance: number or array_like, defaults to None.
        :return: ndarray with shape (n,) if geometry is a scalar.
            Contains tree geometry indices.
        :return: ndarray with shape (2, n) if geometry is an array_like
            The first subarray contains input geometry indices.
            The second subarray contains tree geometry indices.
        """
        return self._str_tree.query(geometry, predicate=predicate, distance=distance)  # type: ignore

    def query_nearest(
        self,
        geometry: Union[BaseGeometry, np.ndarray],
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
        all_matches: bool = True,
    ) -> Union[npt.NDArray[np.int64], Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
        """Queries the str-tree for the nearest geometry to the input geometry.

        :param geometry: The input geometry to query.
        :param max_distance: The maximum distance to consider, defaults to None.
        :param return_distance: Whether to return the distance to the nearest geometry, defaults to False.
        :param exclusive: Whether to exclude the input geometry from the results, defaults to False.
        :param all_matches: Whether to return all matching geometries, defaults to True.
        :return: The nearest geometry or geometries.
        """
        return self._str_tree.query_nearest(
            geometry,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
            all_matches=all_matches,
        )

    def contains_vectorized(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Determines wether input-points are in geometries (i.e. polygons) of the occupancy map.

        Notes
        -----
        This function can be significantly faster than using the str-tree, if the number of geometries is
        relatively small compared to the number of input-points.

        :param points: array of shape (num_points, 2), indexed by :class:`~py123d.geometry.Point2DIndex`.
        :return: boolean array of shape (polygons, input-points)
        """
        output = np.zeros((len(self._geometries), len(points)), dtype=bool)
        for i, geometry in enumerate(self._geometries):
            output[i] = shapely.contains_xy(geometry, points[..., Point2DIndex.X], points[..., Point2DIndex.Y])

        return output
