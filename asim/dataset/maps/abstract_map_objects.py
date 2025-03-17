from __future__ import annotations

import abc


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


class Abstract3DMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: str):
        """
        Constructor of the base map object type.
        :param object_id: unique identifier of the map object.
        """
        self.id = str(object_id)
