from __future__ import annotations

import abc
from typing import Optional, TypeAlias, Union

import numpy as np
import shapely.geometry as geom
import trimesh

from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.geometry import Point3DIndex, Polyline2D, Polyline3D

MapObjectIDType: TypeAlias = Union[str, int]


class BaseMapObject(abc.ABC):

    __slots__ = ("_object_id",)

    def __init__(self, object_id: MapObjectIDType):
        """Constructor of the base map object type.

        :param object_id: unique identifier of the map object.
        """
        self._object_id: MapObjectIDType = object_id

    @property
    def object_id(self) -> MapObjectIDType:
        """Returns the unique identifier of the map object.

        :return: map object id
        """
        return self._object_id

    @property
    @abc.abstractmethod
    def layer(self) -> MapLayer:
        """Returns the map layer type.

        :return: map layer type
        """


class BaseMapSurfaceObject(BaseMapObject):
    """
    Base interface representation of all map objects.
    """

    __slots__ = ("_outline", "_geometry")

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        geometry: Optional[geom.Polygon] = None,
    ) -> None:
        super().__init__(object_id)

        assert outline is not None or geometry is not None, "Either outline or geometry must be provided."

        if outline is None:
            outline = Polyline3D.from_linestring(geometry.exterior)

        if geometry is None:
            geometry = geom.Polygon(outline.array[:, :2])

        self._object_id = object_id
        self._outline = outline
        self._geometry = geometry

    @property
    def outline(self) -> Union[Polyline2D, Polyline3D]:
        return self._outline

    @property
    def outline_2d(self) -> Polyline2D:
        if isinstance(self.outline, Polyline2D):
            return self._outline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return Polyline2D.from_linestring(self._outline.linestring)

    @property
    def outline_3d(self) -> Polyline3D:
        if isinstance(self._outline, Polyline3D):
            return self._outline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self._outline.linestring)

    @property
    def shapely_polygon(self) -> geom.Polygon:
        return self._geometry

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        # Fallback to geometry if no boundaries are available
        outline_3d_array = self.outline_3d.array
        vertices_2d, faces = trimesh.creation.triangulate_polygon(geom.Polygon(outline_3d_array[:, Point3DIndex.XY]))
        if len(vertices_2d) == len(outline_3d_array):
            # Regular case, where vertices match outline_3d_array
            vertices_3d = outline_3d_array
        elif len(vertices_2d) == len(outline_3d_array) + 1:
            # outline array was not closed, so we need to add the first vertex again
            vertices_3d = np.vstack((outline_3d_array, outline_3d_array[0]))
        else:
            raise ValueError("No vertices found for triangulation.")
        trimesh_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
        return trimesh_mesh


class BaseMapLineObject(BaseMapObject):

    __slots__ = ("_polyline",)

    def __init__(self, object_id: MapObjectIDType, polyline: Union[Polyline2D, Polyline3D]) -> None:
        super().__init__(object_id)
        self._polyline = polyline

    @property
    def polyline(self) -> Union[Polyline2D, Polyline3D]:
        return self._polyline

    @property
    def polyline_2d(self) -> Polyline2D:
        if isinstance(self._polyline, Polyline2D):
            return self._polyline
        # Converts 3D polyline to 2D by dropping the z-coordinate
        return Polyline2D.from_linestring(self._polyline.linestring)

    @property
    def polyline_3d(self) -> Polyline3D:
        if isinstance(self._polyline, Polyline3D):
            return self._polyline
        # Converts 2D polyline to 3D by adding a default (zero) z-coordinate
        return Polyline3D.from_linestring(self._polyline.linestring)

    @property
    def shapely_linestring(self) -> geom.LineString:
        return self._polyline.linestring
