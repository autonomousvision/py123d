from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List

from py123d.datatypes import LogMetadata
from py123d.datatypes.metadata.base_metadata import BaseModalityMetadata

if TYPE_CHECKING:
    from py123d.parser.abstract_dataset_parser import FrameData


class AbstractLogWriter(abc.ABC):
    """Abstract base class for log writers.

    A log writer is responsible for specifying the output format of a converted log.
    This includes how data is organized, how it is serialized, and how it is stored.
    """

    @abc.abstractmethod
    def reset(
        self,
        log_metadata: LogMetadata,
        modality_metadatas: List[BaseModalityMetadata],
        deferred_sync: bool = False,
    ) -> bool:
        """Prepare the writer for a new log. Returns True if the log needs writing."""

    def write_sync(self, frame: FrameData) -> None:
        """Write one synchronized frame — all modalities plus one sync-table row."""

    def write_async(self, frame: FrameData, modality_name: str) -> None:
        """Write a single async modality observation from *frame*."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the log writer and finalizes the log io operations."""
