from dataclasses import dataclass

import asim

# TODO: move this files and dataclass to a more appropriate place.


@dataclass
class LogMetadata:

    dataset: str
    log_name: str
    location: str
    timestep_seconds: float

    map_has_z: bool
    version: str = str(asim.__version__)
