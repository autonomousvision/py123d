from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import d123

# TODO: Refactor the usage of the map map metadata in this repo.


@dataclass
class MapMetadata:
    """Class to hold metadata information about a map."""

    dataset: str
    split: Optional[str]  # None, if map is not per log
    log_name: Optional[str]  # None, if map is per log
    location: str
    map_has_z: bool
    map_is_local: bool  # True, if map is per log
    version: str = str(d123.__version__)

    def to_dict(self) -> dict:
        return asdict(self)

    def from_dict(data_dict: Dict[str, Any]) -> MapMetadata:
        return MapMetadata(**data_dict)
