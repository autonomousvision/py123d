from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SceneFilter:

    split_types: List[str] = None
    split_names: List[str] = None
    # scene_tags: List[str] = None
    log_names: Optional[List[str]] = None

    scene_tokens: Optional[List[str]] = None
    map_names: Optional[List[str]] = None
