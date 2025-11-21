from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class IntIDMapping:
    """Class to map string IDs to integer IDs and vice versa."""

    str_to_int: Dict[str, int]

    def __post_init__(self):
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    @classmethod
    def from_series(cls, series: pd.Series) -> IntIDMapping:
        """Creates an IntIDMapping from a pandas Series of string-like IDs."""

        # Drop NaN values and convert all to strings
        unique_ids = series.dropna().astype(str).unique()
        str_to_int = {str_id: idx for idx, str_id in enumerate(unique_ids)}
        return IntIDMapping(str_to_int)

    def map(self, str_like: Any) -> Optional[int]:
        """Maps a string-like ID to its corresponding integer ID."""

        # NOTE: We need to convert a string-like input to an integer ID
        if pd.isna(str_like) or str_like is None:
            return None

        if isinstance(str_like, float):
            key = str(int(str_like))  # Convert float to int first to avoid decimal point
        else:
            key = str(str_like)

        return self.str_to_int.get(key, None)

    def map_list(self, id_list: Optional[List[str]]) -> List[int]:
        """Maps a list of string-like IDs to their corresponding integer IDs."""
        if id_list is None:
            return []
        list_ = []
        for id_str in id_list:
            mapped_id = self.map(id_str)
            if mapped_id is not None:
                list_.append(mapped_id)
        return list_
