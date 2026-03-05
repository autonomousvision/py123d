from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from py123d.datatypes.time.timestamp import Timestamp


class CustomModality:
    """A custom modality for dataset-specific information.

    This class wraps a dictionary (with string keys) and a corresponding
    :class:`~py123d.datatypes.time.Timestamp`. Values can be Python native
    types (``dict``, ``list``, ``str``, ``int``, ``float``, ``bytes``,
    ``bool``, ``None``) or ``numpy.ndarray``.
    """

    __slots__ = ("_data", "_timestamp")

    def __init__(self, data: Dict[str, Any], timestamp: Timestamp) -> None:
        """Initializes a CustomModality instance.

        :param data: The custom data to be stored. Must be a dictionary with string keys.
            Values can be any msgpack-serializable type: ``dict``, ``list``, ``str``,
            ``int``, ``float``, ``bytes``, ``bool``, ``None``, or ``numpy.ndarray``.
        :param timestamp: The :class:`~py123d.datatypes.time.Timestamp` associated with the custom modality data.
        """
        self._data = data
        self._timestamp = timestamp

    @property
    def data(self) -> Dict[str, Any]:
        """The custom data dictionary."""
        return self._data

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of this custom modality."""
        return self._timestamp

    def keys(self) -> List[str]:
        """Returns the keys of the custom data dictionary."""
        return list(self._data.keys())

    def __getitem__(self, key: str) -> Any:
        """Returns the value for *key*. Raises :class:`KeyError` if the key is not present."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Returns ``True`` if *key* exists in the custom data."""
        return key in self._data

    def __len__(self) -> int:
        """Returns the number of entries in the custom data."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterates over the keys of the custom data."""
        return iter(self._data)

    def __getattr__(self, name: str) -> Optional[Any]:
        """Provides attribute-style access to data keys. Returns ``None`` if the key is not present."""
        attr: Optional[Any] = self._data[name] if name in self._data.keys() else None
        return attr

    def __repr__(self) -> str:
        """Returns a string representation showing the available keys and timestamp."""
        keys = list(self._data.keys())
        return f"CustomModality(keys={keys}, timestamp={self._timestamp})"
