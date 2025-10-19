from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class IntIDMapping:

    str_to_int: Dict[str, int]

    def __post_init__(self):
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    @classmethod
    def from_series(cls, series: pd.Series) -> IntIDMapping:
        unique_ids = series.unique()
        str_to_int = {str_id: idx for idx, str_id in enumerate(unique_ids)}
        return IntIDMapping(str_to_int)

    def map_list(self, id_list: Optional[List[str]]) -> pd.Series:
        if id_list is None:
            return []
        return [self.str_to_int.get(id_str, -1) for id_str in id_list]


class IncrementalIntIDMapping:

    def __init__(self):
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.next_id: int = 0

    def get_int_id(self, str_id: str) -> int:
        if str_id not in self.str_to_int:
            self.str_to_int[str_id] = self.next_id
            self.int_to_str[self.next_id] = str_id
            self.next_id += 1
        return self.str_to_int[str_id]
