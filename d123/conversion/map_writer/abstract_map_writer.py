import abc
from abc import abstractmethod

from d123.conversion.dataset_converter_config import DatasetConverterConfig
from d123.datatypes.maps.abstract_map_objects import (
    AbstractCarpark,
    AbstractCrosswalk,
    AbstractGenericDrivable,
    AbstractIntersection,
    AbstractLane,
    AbstractLaneGroup,
    AbstractRoadEdge,
    AbstractRoadLine,
    AbstractStopLine,
    AbstractWalkway,
)
from d123.datatypes.scene.scene_metadata import LogMetadata


class AbstractMapWriter(abc.ABC):
    """Abstract base class for map writers."""

    @abstractmethod
    def reset(self, dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> bool:
        """Reset the writer to its initial state."""

    @abstractmethod
    def write_lane(self, lane: AbstractLane) -> None:
        """Write a lane to the map."""

    @abstractmethod
    def write_lane_group(self, lane: AbstractLaneGroup) -> None:
        """Write a group of lanes to the map."""

    @abstractmethod
    def write_intersection(self, intersection: AbstractIntersection) -> None:
        """Write an intersection to the map."""

    @abstractmethod
    def write_crosswalk(self, crosswalk: AbstractCrosswalk) -> None:
        """Write a crosswalk to the map."""

    @abstractmethod
    def write_carpark(self, carpark: AbstractCarpark) -> None:
        """Write a car park to the map."""

    @abstractmethod
    def write_walkway(self, walkway: AbstractWalkway) -> None:
        """Write a walkway to the map."""

    @abstractmethod
    def write_generic_drivable(self, obj: AbstractGenericDrivable) -> None:
        """Write a generic drivable area to the map."""

    @abstractmethod
    def write_stop_line(self, stop_line: AbstractStopLine) -> None:
        """Write a stop lines to the map."""

    @abstractmethod
    def write_road_edge(self, road_edge: AbstractRoadEdge) -> None:
        """Write a road edge to the map."""

    @abstractmethod
    def write_road_line(self, road_line: AbstractRoadLine) -> None:
        """Write a road line to the map."""

    @abstractmethod
    def close(self) -> None:
        """Close the writer and finalize any resources."""
