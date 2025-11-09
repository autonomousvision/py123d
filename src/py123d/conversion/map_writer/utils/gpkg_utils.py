from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class IntIDMapping:

    str_to_int: Dict[str, int]

    def __post_init__(self):
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    @classmethod
    def from_series(cls, series: pd.Series) -> IntIDMapping:
        # Drop NaN values and convert all to strings
        unique_ids = series.dropna().astype(str).unique()
        str_to_int = {str_id: idx for idx, str_id in enumerate(unique_ids)}
        return IntIDMapping(str_to_int)

    def map(self, str_like: Any) -> Optional[int]:
        # NOTE: We need to convert a string-like input to an integer ID
        if pd.isna(str_like) or str_like is None:
            return None

        if isinstance(str_like, float):
            key = str(int(str_like))  # Convert float to int first to avoid decimal point
        else:
            key = str(str_like)

        return self.str_to_int.get(key, None)

    def map_list(self, id_list: Optional[List[str]]) -> List[int]:
        if id_list is None:
            return []
        list_ = []
        for id_str in id_list:
            mapped_id = self.map(id_str)
            if mapped_id is not None:
                list_.append(mapped_id)
        return list_


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
