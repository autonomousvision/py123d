from __future__ import annotations

import abc
from typing import Dict, Iterable, List, Optional, Union

import shapely

from d123.dataset.maps.abstract_map_objects import AbstractMapObject
from d123.dataset.maps.map_datatypes import MapLayer
from d123.geometry import Point2D

# TODO:
# - add docstrings
# - rename methods?
# - Combine query and query_object_ids into one method with an additional parameter to specify whether to return objects or IDs?
# - Add stop pads or stop lines.


class AbstractMap(abc.ABC):

    @property
    @abc.abstractmethod
    def map_name(self) -> str:
        pass

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def get_available_map_objects(self) -> List[MapLayer]:
        pass

    @abc.abstractmethod
    def get_map_object(self, object_id: str, layer: MapLayer) -> Optional[AbstractMapObject]:
        pass

    @abc.abstractmethod
    def get_all_map_objects(self, point_2d: Point2D, layer: MapLayer) -> List[AbstractMapObject]:
        pass

    @abc.abstractmethod
    def is_in_layer(self, point: Point2D, layer: MapLayer) -> bool:
        pass

    @abc.abstractmethod
    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[MapLayer]
    ) -> Dict[MapLayer, List[AbstractMapObject]]:
        pass

    @abc.abstractmethod
    def query(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]]:
        pass

    @abc.abstractmethod
    def query_object_ids(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[str], Dict[int, List[str]]]]:
        pass

    @abc.abstractmethod
    def query_nearest(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        return_all: bool = True,
        max_distance: Optional[float] = None,
        return_distance: bool = False,
        exclusive: bool = False,
    ) -> Dict[MapLayer, Union[List[AbstractMapObject], Dict[int, List[AbstractMapObject]]]]:
        pass
