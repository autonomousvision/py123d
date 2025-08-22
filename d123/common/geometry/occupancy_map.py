from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import shapely.vectorized
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

# TODO: Figure out if a 3D equivalent is needed.


class OccupancyMap2D:
    def __init__(
        self,
        geometries: Sequence[BaseGeometry],
        ids: Optional[Union[List[str], List[int]]] = None,
        node_capacity: int = 10,
    ):
        """
        Constructor of PDMOccupancyMap
        :param geometries: list/array of polygons
        :param ids: optional list of geometry identifiers
        :param node_capacity: max number of child nodes in str-tree, defaults to 10
        """
        assert ids is None or len(ids) == len(geometries), "Length of ids must match length of geometries"
        # assert len(tokens) == len(geometries)

        self._ids: Union[List[str], List[int]] = (
            ids if ids is not None else [str(idx) for idx in range(len(geometries))]
        )
        self._id_to_idx: Dict[Union[str, int], int] = {id: idx for idx, id in enumerate(self._ids)}

        self._geometries = geometries
        self._node_capacity = node_capacity
        self._str_tree = STRtree(self._geometries, node_capacity)

    @classmethod
    def from_dict(cls, geometry_dict: Dict[Union[str, int], BaseGeometry], node_capacity: int = 10) -> OccupancyMap2D:
        ids = list(geometry_dict.keys())
        geometries = list(geometry_dict.values())
        return cls(geometries=geometries, ids=ids, node_capacity=node_capacity)

    def __getitem__(self, id: Union[str, int]) -> BaseGeometry:
        """
        Retrieves geometry of token.
        :param token: geometry identifier
        :return: Geometry of token
        """
        return self._geometries[self._id_to_idx[id]]

    def __len__(self) -> int:
        """
        Number of geometries in the occupancy map
        :return: int
        """
        return len(self._ids)

    @property
    def ids(self) -> Union[List[str], List[int]]:
        """
        Getter for track tokens in occupancy map
        :return: list of strings
        """
        return self._ids

    @property
    def geometries(self) -> Sequence[BaseGeometry]:

        return self._geometries

    @property
    def token_to_idx(self) -> Dict[Union[int, str], int]:
        """
        Getter for track tokens in occupancy map
        :return: dictionary of tokens and indices
        """
        return self._id_to_idx

    def intersects(self, geometry: BaseGeometry) -> Union[List[str], List[int]]:
        """
        Searches for intersecting geometries in the occupancy map
        :param geometry: geometries to query
        :return: list of tokens for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._ids[idx] for idx in indices]

    def query(
        self,
        geometry: Union[BaseGeometry, np.ndarray],
        predicate: Optional[
            Literal["intersects", "within", "contains", "overlaps", "crosses", "touches", "covers", "covered_by"]
        ] = None,
        distance: Optional[float] = None,
    ) -> npt.NDArray[np.int64]:
        """
        Function to directly calls shapely's query function on str-tree
        :param geometry: geometries to query
        :param predicate: see shapely, defaults to None
        :return: query output
        """
        return self._str_tree.query(geometry, predicate=predicate, distance=distance)

    def query_nearest(
        self,
        geometry: Union[BaseGeometry, np.ndarray],
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
        all_matches: bool = True,
    ):
        return self._str_tree.query_nearest(
            geometry,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
            all_matches=all_matches,
        )

    def points_in_polygons(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Determines wether input-points are in polygons of the occupancy map
        :param points: input-points
        :return: boolean array of shape (polygons, input-points)
        """
        output = np.zeros((len(self._geometries), len(points)), dtype=bool)
        for i, polygon in enumerate(self._geometries):
            output[i] = shapely.vectorized.contains(polygon, points[:, 0], points[:, 1])

        return output
