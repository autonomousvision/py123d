from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SceneFilter:

    split_types: List[str] = None
    split_names: List[str] = None
    # scene_tags: List[str] = None
    log_names: Optional[List[str]] = None

    map_names: Optional[List[str]] = None  # TODO:
    scene_tokens: Optional[List[str]] = None  # TODO:

    timestamp_threshold_s: Optional[float] = None  # TODO:
    ego_displacement_minimum_m: Optional[float] = None  # TODO:

    duration_s: Optional[float] = 10.0
    history_s: Optional[float] = 3.0
