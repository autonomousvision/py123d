from __future__ import annotations

import abc
from typing import Dict, Iterable, List, Optional, Union

import shapely

from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry import Point2D

# TODO:
# - add docstrings
# - rename methods?
# - Combine query and query_object_ids into one method with an additional parameter to specify whether to return objects or IDs?
# - Add stop pads or stop lines.


class MapAPI(abc.ABC):

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        pass

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def get_available_map_objects(self) -> List[MapLayer]:
        pass

    @abc.abstractmethod
    def get_map_object(self, object_id: str, layer: MapLayer) -> Optional[BaseMapObject]:
        pass

    @abc.abstractmethod
    def get_all_map_objects(self, point_2d: Point2D, layer: MapLayer) -> List[BaseMapObject]:
        pass

    @abc.abstractmethod
    def is_in_layer(self, point: Point2D, layer: MapLayer) -> bool:
        pass

    @abc.abstractmethod
    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[MapLayer]
    ) -> Dict[MapLayer, List[BaseMapObject]]:
        pass

    @abc.abstractmethod
    def query(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]]:
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
    ) -> Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]]:
        pass

    @property
    def location(self) -> str:
        return self.get_map_metadata().location
