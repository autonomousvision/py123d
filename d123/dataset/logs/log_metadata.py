from dataclasses import dataclass

import d123

# TODO: move this files and dataclass to a more appropriate place.


@dataclass
class LogMetadata:

    # TODO: add
    # - split
    # - global/local map

    dataset: str
    log_name: str
    location: str
    timestep_seconds: float

    map_has_z: bool
    version: str = str(d123.__version__)
