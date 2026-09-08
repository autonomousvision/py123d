"""Map parser for the Griffin dataset.

Griffin is rendered in CARLA on four stock towns (Town03, Town06, Town07,
Town10HD) whose OpenDRIVE definitions are already bundled with py123d for the
:class:`~py123d.parser.opendrive.opendrive_parser.OpenDriveParser`. Griffin's
global (ENU) frame coincides with the CARLA/OpenDRIVE map frame (identity
transform), so map support reduces to re-exporting the bundled town maps under
the ``griffin`` dataset with the town as ``location`` — mirroring the global-map
linkage used by nuScenes (``map_is_per_log=False``; logs resolve their map via
``{maps_root}/griffin/griffin_{town}.arrow``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Tuple

from typing_extensions import override

from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.opendrive.opendrive_map_parser import OpenDriveMapParser

_CARLA_MAPS_DIR: Final[Path] = Path(__file__).parents[1] / "opendrive" / "carla_maps"

GRIFFIN_TOWNS: Final[Tuple[str, ...]] = ("Town03", "Town06", "Town07", "Town10HD")
"""CARLA towns used by the official Griffin release."""


class GriffinMapParser(OpenDriveMapParser):
    """Map parser for one Griffin (CARLA) town.

    Thin subclass of :class:`~py123d.parser.opendrive.opendrive_map_parser.OpenDriveMapParser`
    over the bundled CARLA ``.xodr.gz`` files; only the map metadata is
    re-branded so the maps are written under the ``griffin`` dataset and picked
    up by Griffin logs via their ``location``.
    """

    def __init__(self, town: str) -> None:
        """Initialize the :class:`GriffinMapParser`.

        :param town: CARLA town name, e.g. ``"Town03"``. Must be one of
            :data:`GRIFFIN_TOWNS`.
        """
        assert town in GRIFFIN_TOWNS, f"Town {town} is not part of Griffin. Available towns: {GRIFFIN_TOWNS}"
        self._town = town
        super().__init__(xodr_path=_CARLA_MAPS_DIR / f"{town}.xodr.gz", location=town)

    @override
    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return griffin_map_metadata(self._town)


def griffin_map_metadata(town: str) -> MapMetadata:
    """Build the global :class:`MapMetadata` for a Griffin ``town``.

    Single source of truth shared by :class:`GriffinMapParser` and the log
    parsers (which attach it to ``LogMetadata`` so ``has_map`` / ``map_locations``
    scene filters resolve correctly).

    :param town: CARLA town name, e.g. ``"Town03"``.
    :return: Global map metadata (``dataset="griffin"``, ``map_is_per_log=False``).
    """
    return MapMetadata(dataset="griffin", location=town, map_has_z=True, map_is_per_log=False)


def get_griffin_map_parsers() -> List[BaseMapParser]:
    """Return one :class:`GriffinMapParser` per Griffin town.

    Shared by the vehicle- and drone-side dataset parsers; the map writer skips
    towns that were already converted, so running both conversions is safe.

    :return: List of map parsers covering :data:`GRIFFIN_TOWNS`.
    """
    return [GriffinMapParser(town) for town in GRIFFIN_TOWNS]
