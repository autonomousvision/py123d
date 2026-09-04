from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from py123d.datatypes.metadata.base_metadata import BaseMetadata


@dataclass(frozen=True)
class SourceModality(BaseMetadata):
    """One modality of the same log that a derived modality was produced from.

    :param modality_key: Key of the source modality, e.g. ``"ego_state_se3"``.
    :param columns: Arrow column names the producer consumed, without the modality prefix.
    :param digest: Fingerprint of those columns as returned by ``hash_modality_columns``.
    """

    modality_key: str
    columns: List[str]
    digest: str

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> SourceModality:
        """Inherited, see superclass."""
        return cls(
            modality_key=data_dict["modality_key"],
            columns=[str(column) for column in data_dict["columns"]],
            digest=data_dict["digest"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {"modality_key": self.modality_key, "columns": list(self.columns), "digest": self.digest}


@dataclass(frozen=True)
class Provenance(BaseMetadata):
    """Where a derived modality came from, stored in its modality metadata.

    A derived modality (the route, an autolabel, a backfilled stream) is a cache of a
    computation over other data. Without a key such a cache cannot tell that its inputs
    have changed. This record is that key: it names the producer and fingerprints every
    source modality the producer read from the same log, so a reader can re-hash those
    sources and detect staleness.

    Sources that live outside the log (an HD map, a routing service, a model checkpoint)
    cannot be re-hashed by py123d. Producers that depend on them state them in
    ``external_sources``, which is reported but never verified.

    :param producer: Namespaced producer identifier with a version, e.g.
        ``"py123d:route_from_odometry@1"`` or ``"garage:route_backfill@3"``.
    :param source_modalities: One entry per modality of the same log the producer consumed.
    :param produced_at: Wall-clock time of the computation as an ISO-8601 UTC string.
    :param external_sources: Free-form descriptions of sources outside the log. These are
        never verified.
    """

    producer: str
    source_modalities: List[SourceModality]
    produced_at: str
    external_sources: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert ":" in self.producer and "@" in self.producer, (
            f"producer must be '<namespace>:<producer>@<version>', got {self.producer!r}."
        )

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> Provenance:
        """Inherited, see superclass."""
        return cls(
            producer=data_dict["producer"],
            source_modalities=[SourceModality.from_dict(entry) for entry in data_dict["source_modalities"]],
            produced_at=data_dict["produced_at"],
            external_sources=data_dict.get("external_sources"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {
            "producer": self.producer,
            "source_modalities": [source.to_dict() for source in self.source_modalities],
            "produced_at": self.produced_at,
            "external_sources": self.external_sources,
        }
