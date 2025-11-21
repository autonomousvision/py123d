from __future__ import annotations

import abc
from typing import Dict, Iterable, List, Literal, Optional, Union

import shapely

from py123d.datatypes.map_objects import BaseMapObject, MapLayer
from py123d.datatypes.metadata import MapMetadata
from py123d.geometry import Point2D
from py123d.geometry.point import Point3D


class MapAPI(abc.ABC):
    """The base class for all map APIs in 123D."""

    # Abstract Methods, to be implemented by subclasses
    # ------------------------------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def get_map_metadata(self) -> MapMetadata:
        """Returns the :class:`~p123d.datatypes.metadata.MapMetadata` of the map api.

        :return: The map metadata, e.g. location, dataset, etc.
        """

    @abc.abstractmethod
    def get_available_map_layers(self) -> List[MapLayer]:
        """Returns the available :class:`~p123d.datatypes.map_objects.map_layer_types.MapLayer`,
            e.g. LANE, LANE_GROUP, etc.

        :return: A list of available map layers.
        """

    @abc.abstractmethod
    def get_map_object(self, object_id: Union[str, int], layer: MapLayer) -> Optional[BaseMapObject]:
        """Returns a :class:`~p123d.datatypes.map_objects.base_map_object.BaseMapObject` by its ID
            and :class:`~p123d.datatypes.map_objects.map_layer_types.MapLayer`.

        :param object_id: The ID of the map object.
        :param layer: The layer the map object belongs to.
        :return: The map object if found, None otherwise.
        """

    @abc.abstractmethod
    def get_map_objects_in_radius(
        self,
        point: Union[Point2D, Point3D],
        radius: float,
        layers: List[MapLayer],
    ) -> Dict[MapLayer, List[BaseMapObject]]:
        """Returns a dictionary of :class:`~p123d.datatypes.map_objects.map_layer_types.MapLayer` to a list of
            :class:`~p123d.datatypes.map_objects.base_map_object.BaseMapObject` within a given radius
            around a center point.

        :param point: The center point to search around.
        :param radius: The radius to search within.
        :param layers: The map layers to search in.
        :return: A dictionary mapping each layer to a list of map objects within the radius.
        """

    @abc.abstractmethod
    def query(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[
            Literal[
                "contains",
                "contains_properly",
                "covered_by",
                "covers",
                "crosses",
                "intersects",
                "overlaps",
                "touches",
                "within",
                "dwithin",
            ]
        ] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[BaseMapObject], Dict[int, List[BaseMapObject]]]]:
        """Queries geometries against the map objects in the specified layers using an optional spatial predicate.

        Notes
        -----
        The implementation is aligned with the geopandas spatial index query method [1]_,
        used in the :class:`GPKGMapAPI`. It is likely this method will be removed or improved in
        future versions of 123D.

        References
        ----------
        [1] https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query.html

        :param geometry: A shapely geometry or an iterable of shapely geometries to query against.
        :param layers: The map layers to query against.
        :param predicate: An optional spatial predicate to filter the results.
        :param sort: Whether to sort the results by distance, defaults to False.
        :param distance: An optional maximum distance to filter the results, defaults to None.
        :return: A dictionary mapping each layer to a list of map objects or a dictionary of indices to
            lists of map objects.
        """

    @abc.abstractmethod
    def query_object_ids(
        self,
        geometry: Union[shapely.Geometry, Iterable[shapely.Geometry]],
        layers: List[MapLayer],
        predicate: Optional[str] = None,
        sort: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[MapLayer, Union[List[int], Dict[int, List[int]]]]:
        """Queries geometries against the map objects in the specified layers using an optional spatial predicate.
        Instead of returning the map objects, it returns their IDs only.

        Notes
        -----
        The implementation is aligned with the geopandas spatial index query method [1]_,
        used in the :class:`GPKGMapAPI`. It is likely this method will be removed or improved in
        future versions of 123D.

        References
        ----------
        [1] https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query.html


        :param geometry: A shapely geometry or an iterable of shapely geometries to query against.
        :param layers: The map layers to query against.
        :param predicate: An optional spatial predicate to filter the results.
        :param sort: Whether to sort the results by distance, defaults to False.
        :param distance: An optional maximum distance to filter the results, defaults to None.
        :return: A dictionary mapping each layer to a list of map object IDs or a dictionary of indices to
            lists of map object IDs.
        """

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
        """Return the nearest map objects in a spatial tree for each input geometry in

        Notes
        -----
        The implementation is aligned with the geopandas spatial index nearest method [1]_,
        used in the :class:`GPKGMapAPI`. It is likely this method will be removed or improved in
        future versions of 123D.

        References
        ----------
        [1] https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.nearest.html

        :param geometry: A shapely geometry or an iterable of shapely geometries to query against.
        :param layers: The map layers to query against.
        :param return_all: Whether to return all matching objects or just the closest one, defaults to True.
        :param max_distance: An optional maximum distance to filter the results, defaults to None.
        :param return_distance: Whether to return the distance to the nearest object, defaults to False.
        :param exclusive: Whether to exclude the input geometries from the results, defaults to False.
        :return: A dictionary mapping each layer to a list of nearest map objects or a dictionary of indices to
            lists of nearest map objects.
        """

    # Syntactic Sugar / Properties, for easier access to common attributes
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def map_metadata(self) -> MapMetadata:
        """The :class:`~py123d.datatypes.metadata.MapMetadata` of the map api."""
        return self.get_map_metadata()

    @property
    def dataset(self) -> str:
        """The dataset name from the map metadata."""
        return self.map_metadata.dataset

    @property
    def location(self) -> str:
        """The location from the map metadata."""
        return self.map_metadata.location

    @property
    def map_is_local(self) -> bool:
        """Indicates if the map is local (map for each log) or global (map for multiple logs in dataset)."""
        return self.map_metadata.map_is_local

    @property
    def map_has_z(self) -> bool:
        """Indicates if the map includes Z (elevation) data."""
        return self.map_metadata.map_has_z

    @property
    def version(self) -> str:
        """The version of the py123d library used to create this map metadata."""
        return self.map_metadata.version
