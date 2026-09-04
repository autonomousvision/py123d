"""Producing and verifying the :class:`Provenance` record of derived modalities.

A derived modality is written once and read many times, so its source modalities are
fingerprinted at write time and re-fingerprinted when the log is opened. The fingerprint
covers only the columns the producer actually consumed, so an unrelated fix elsewhere in
the source modality does not invalidate the derived data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa

from py123d.datatypes.metadata.provenance import Provenance, SourceModality
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata

_HASH_DIGEST_SIZE = 16


class StaleModalityError(ValueError):
    """Raised when a derived modality's source modalities no longer match their recorded fingerprints."""


def utc_now_iso() -> str:
    """The current wall-clock time as an ISO-8601 UTC string, e.g. ``2026-08-31T09:14:22Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_provenance(modality_metadata: BaseModalityMetadata) -> Optional[Provenance]:
    """The :class:`Provenance` record of a modality metadata, or None for raw modalities."""
    provenance = getattr(modality_metadata, "provenance", None)
    return provenance if isinstance(provenance, Provenance) else None


def _column_bytes(column: pa.ChunkedArray, qualified_name: str) -> bytes:
    """The contiguous value bytes of one Arrow column.

    Only primitive and fixed-size-list-of-primitive columns are supported, which covers
    the numeric streams derived modalities consume. Anything else raises, so a producer
    learns at write time that its source cannot be fingerprinted.
    """
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        array = array.flatten()
    if not (pa.types.is_primitive(array.type) or pa.types.is_boolean(array.type)):
        raise TypeError(
            f"Cannot fingerprint column '{qualified_name}' of type {array.type}: "
            "only primitive and fixed-size-list-of-primitive columns are supported."
        )
    return array.to_numpy(zero_copy_only=False).tobytes()


def hash_modality_columns(log_dir: Path, modality_key: str, columns: Sequence[str]) -> str:
    """Fingerprint the given columns of one modality file in a log directory.

    :param log_dir: The log directory holding ``{modality_key}.arrow``.
    :param modality_key: The modality key, e.g. ``"ego_state_se3"``.
    :param columns: Column names without the modality prefix, e.g. ``["imu_se3"]``.
    :return: A hex digest over the columns' values, in the given order.
    :raises FileNotFoundError: If the modality file does not exist.
    """
    from py123d.api.utils.arrow_helper import open_arrow_table

    path = Path(log_dir) / f"{modality_key}.arrow"
    if not path.exists():
        raise FileNotFoundError(f"Cannot fingerprint '{modality_key}': {path} does not exist.")

    table = open_arrow_table(path)
    digest = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    for column in columns:
        qualified_name = f"{modality_key}.{column}"
        digest.update(qualified_name.encode())
        digest.update(_column_bytes(table.column(qualified_name), qualified_name))
    return digest.hexdigest()


def build_provenance(
    log_dir: Path,
    producer: str,
    source_columns: Dict[str, Sequence[str]],
    external_sources: Optional[List[str]] = None,
) -> Provenance:
    """Fingerprint every consumed source modality and stamp the record for a derived modality.

    :param log_dir: The log directory the source modalities are read from.
    :param producer: Namespaced producer identifier, e.g. ``"garage:route_backfill@3"``.
    :param source_columns: Mapping of source modality key to the columns the producer consumed.
    :param external_sources: Descriptions of sources outside the log, which stay unverified.
    """
    source_modalities = [
        SourceModality(
            modality_key=modality_key,
            columns=list(columns),
            digest=hash_modality_columns(log_dir, modality_key, columns),
        )
        for modality_key, columns in sorted(source_columns.items())
    ]
    return Provenance(
        producer=producer,
        source_modalities=source_modalities,
        produced_at=utc_now_iso(),
        external_sources=external_sources,
    )


@lru_cache(maxsize=10_000)
def verify_log_consistency(log_dir: Path) -> None:
    """Re-fingerprint the declared source modalities of every derived modality in a log directory.

    Modalities without a :class:`Provenance` record are raw and skipped. The result is
    cached per process, so a log is checked once no matter how often it is opened.

    Staleness is the only condition raised here. A log that cannot be read at all, or
    whose declared source is missing, is a broken log rather than a stale one, and is left
    to the caller's own error handling.

    :param log_dir: The log directory to check.
    :raises StaleModalityError: If any source modality no longer matches its recorded fingerprint.
    """
    from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata

    log_dir = Path(log_dir)
    try:
        modality_metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
    except Exception:
        return

    for modality_key, modality_metadata in modality_metadatas.items():
        provenance = get_provenance(modality_metadata)
        if provenance is None:
            continue
        for source in provenance.source_modalities:
            try:
                actual_digest = hash_modality_columns(log_dir, source.modality_key, source.columns)
            except (FileNotFoundError, KeyError, TypeError):
                continue
            if actual_digest != source.digest:
                raise StaleModalityError(
                    f"'{modality_key}' in {log_dir} is stale: its source '{source.modality_key}' "
                    f"(columns {', '.join(source.columns)}) hashes to {actual_digest} but was {source.digest} when "
                    f"'{provenance.producer}' produced it on {provenance.produced_at}. "
                    f"Recompute '{modality_key}' with '{provenance.producer}'."
                )
