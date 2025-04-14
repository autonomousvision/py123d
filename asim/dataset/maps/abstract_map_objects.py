from __future__ import annotations

import abc
from typing import List

import shapely.geometry as geom


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


class AbstractLane(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base lane type.
        :param object_id: unique identifier of the lane.
        """
        super().__init__(object_id)

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

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass


class AbstractLaneGroup(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base lane group type.
        :param object_id: unique identifier of the lane group.
        """
        super().__init__(object_id)

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass


class AbstractIntersection(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base intersection type.
        :param object_id: unique identifier of the intersection.
        """
        super().__init__(object_id)

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass


class AbstractCrosswalk(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base crosswalk type.
        :param object_id: unique identifier of the crosswalk.
        """
        super().__init__(object_id)

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass


class AbstractCarpark(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base carpark type.
        :param object_id: unique identifier of the carpark.
        """
        super().__init__(object_id)

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass


class AbstractGenericDrivable(AbstractMapObject):
    def __init__(self, object_id: str):
        """
        Constructor of the base generic drivable area type.
        :param object_id: unique identifier of the generic drivable area.
        """
        super().__init__(object_id)

    @property
    @abc.abstractmethod
    def shapely_polygon(self) -> geom.Polygon:
        pass
