from __future__ import annotations

import abc
from typing import List

import shapely.geometry as geom

from asim.dataset.maps.map_datatypes import MapSurfaceType


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: str):
        """
        Constructor of the base map object type.
        :param object_id: unique identifier of the map object.
        """
        self.id = str(object_id)


class AbstractSurfaceMapObject(AbstractMapObject):
    """
    Base interface representation of all map objects.
    """

    # pass
    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass

    @property
    @abc.abstractmethod
    def surface_type(self) -> MapSurfaceType:
        pass

    # TODO: implement
    # @property
    # @abc.abstractmethod
    # def outline_3d(self) -> Polyline3D:
    #     pass

    # @property
    # @abc.abstractmethod
    # def outline_2d(self) -> Polyline2D:
    #     pass

    # @property
    # @abc.abstractmethod
    # def trimesh(self) -> ...:
    #     pass


class AbstractLane(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.LANE

    @property
    @abc.abstractmethod
    def successors(self) -> List[AbstractLane]:
        """
        Property of succeeding lane objects (front).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def predecessors(self) -> List[AbstractLane]:
        """
        Property of preceding lane objects (behind).
        :return: list of lane class
        """


class AbstractLaneGroup(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.LANE_GROUP

    @property
    @abc.abstractmethod
    def successors(self) -> List[AbstractLaneGroup]:
        """
        Property of succeeding lane objects (front).
        :return: list of lane class
        """

    @property
    @abc.abstractmethod
    def predecessors(self) -> List[AbstractLaneGroup]:
        """
        Property of preceding lane objects (behind).
        :return: list of lane class
        """


class AbstractIntersection(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.INTERSECTION


class AbstractCrosswalk(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.CROSSWALK


class AbstractWalkway(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.WALKWAY


class AbstractCarpark(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.CARPARK


class AbstractGenericDrivable(AbstractSurfaceMapObject):
    @property
    def surface_type(self) -> MapSurfaceType:
        return MapSurfaceType.GENERIC_DRIVABLE
