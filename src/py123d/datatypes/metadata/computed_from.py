from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from py123d.datatypes.metadata.base_metadata import BaseMetadata


@dataclass(frozen=True)
class ComputedFrom(BaseMetadata):
    """Record of how a derived modality was produced, stored in its modality metadata.

    A derived modality (the route, an autolabel, a backfilled stream) is a cache of a
    computation over other data. Without a key such a cache cannot tell that its inputs
    have changed. This record is that key: it names the producer and fingerprints every
    input the producer read from the same log, so a reader can re-hash those inputs and
    detect staleness.

    Inputs that live outside the log (an HD map, a routing service, a model checkpoint)
    cannot be re-hashed by py123d. Producers that depend on them state them in
    ``external_inputs``, which is reported but never verified.

    :param computed_by: Namespaced producer identifier with a version, e.g.
        ``"py123d:route_from_odometry@1"`` or ``"garage:route_backfill@3"``.
    :param input_hashes: One ``(modality_key, columns, hash)`` triple per consumed input,
        where ``columns`` is a comma-separated list of Arrow column names without the
        modality prefix, and ``hash`` is the digest returned by ``hash_modality_columns``.
    :param computed_at: Wall-clock time of the computation as an ISO-8601 UTC string.
    :param external_inputs: Free-form descriptions of inputs outside the log. These are
        never verified.
    """

    computed_by: str
    input_hashes: List[Tuple[str, str, str]]
    computed_at: str
    external_inputs: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert ":" in self.computed_by and "@" in self.computed_by, (
            f"computed_by must be '<namespace>:<producer>@<version>', got {self.computed_by!r}."
        )

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> ComputedFrom:
        """Inherited, see superclass."""
        return cls(
            computed_by=data_dict["computed_by"],
            input_hashes=[(str(key), str(columns), str(digest)) for key, columns, digest in data_dict["input_hashes"]],
            computed_at=data_dict["computed_at"],
            external_inputs=data_dict.get("external_inputs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {
            "computed_by": self.computed_by,
            "input_hashes": [list(entry) for entry in self.input_hashes],
            "computed_at": self.computed_at,
            "external_inputs": self.external_inputs,
        }
