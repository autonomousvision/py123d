"""Producing and verifying the :class:`ComputedFrom` record of derived modalities.

A derived modality is written once and read many times, so its inputs are fingerprinted
at write time and re-fingerprinted when the log is opened. The fingerprint covers only
the columns the producer actually consumed, so an unrelated fix elsewhere in the input
modality does not invalidate the derived data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa

from py123d.datatypes.metadata.computed_from import ComputedFrom
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata

_HASH_DIGEST_SIZE = 16


class StaleModalityError(ValueError):
    """Raised when a derived modality's inputs no longer match its recorded fingerprint."""


def utc_now_iso() -> str:
    """The current wall-clock time as an ISO-8601 UTC string, e.g. ``2026-08-31T09:14:22Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_computed_from(modality_metadata: BaseModalityMetadata) -> Optional[ComputedFrom]:
    """The :class:`ComputedFrom` record of a modality metadata, or None for raw modalities."""
    computed_from = getattr(modality_metadata, "computed_from", None)
    return computed_from if isinstance(computed_from, ComputedFrom) else None


def _column_bytes(column: pa.ChunkedArray, qualified_name: str) -> bytes:
    """The contiguous value bytes of one Arrow column.

    Only primitive and fixed-size-list-of-primitive columns are supported, which covers
    the numeric streams derived modalities consume. Anything else raises, so a producer
    learns at write time that its input cannot be fingerprinted.
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


def build_computed_from(
    log_dir: Path,
    computed_by: str,
    inputs: Dict[str, Sequence[str]],
    external_inputs: Optional[List[str]] = None,
) -> ComputedFrom:
    """Fingerprint every consumed input and stamp the record for a derived modality.

    :param log_dir: The log directory the inputs are read from.
    :param computed_by: Namespaced producer identifier, e.g. ``"garage:route_backfill@3"``.
    :param inputs: Mapping of modality key to the columns the producer consumed.
    :param external_inputs: Descriptions of inputs outside the log, which stay unverified.
    """
    input_hashes = [
        (modality_key, ",".join(columns), hash_modality_columns(log_dir, modality_key, columns))
        for modality_key, columns in sorted(inputs.items())
    ]
    return ComputedFrom(
        computed_by=computed_by,
        input_hashes=input_hashes,
        computed_at=utc_now_iso(),
        external_inputs=external_inputs,
    )


@lru_cache(maxsize=10_000)
def verify_log_consistency(log_dir: Path) -> None:
    """Re-fingerprint the declared inputs of every derived modality in a log directory.

    Modalities without a :class:`ComputedFrom` record are raw and skipped. The result is
    cached per process, so a log is checked once no matter how often it is opened.

    Staleness is the only condition raised here. A log that cannot be read at all, or
    whose declared input is missing, is a broken log rather than a stale one, and is left
    to the caller's own error handling.

    :param log_dir: The log directory to check.
    :raises StaleModalityError: If any input no longer matches its recorded fingerprint.
    """
    from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata

    log_dir = Path(log_dir)
    try:
        modality_metadatas = parse_log_directory_metadata(log_dir).modality_metadatas
    except Exception:
        return

    for modality_key, modality_metadata in modality_metadatas.items():
        computed_from = get_computed_from(modality_metadata)
        if computed_from is None:
            continue
        for input_key, columns, expected_hash in computed_from.input_hashes:
            try:
                actual_hash = hash_modality_columns(log_dir, input_key, columns.split(","))
            except (FileNotFoundError, KeyError, TypeError):
                continue
            if actual_hash != expected_hash:
                raise StaleModalityError(
                    f"'{modality_key}' in {log_dir} is stale: its input '{input_key}.{columns}' hashes to "
                    f"{actual_hash} but was {expected_hash} when '{computed_from.computed_by}' wrote it on "
                    f"{computed_from.computed_at}. Recompute '{modality_key}' with '{computed_from.computed_by}'."
                )
