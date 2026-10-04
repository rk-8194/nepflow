"""Encode, decode, and identify records stored in the SQLite ledger."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from nepflow.domain.identities import DftCalculationIdentity
from nepflow.errors import StateError
from nepflow.io.json import dumps, loads, to_jsonable


def now() -> str:
    """Return the ledger's canonical UTC timestamp representation."""

    return datetime.now(timezone.utc).isoformat()


def encode_json(value: Any) -> str:
    """Encode a ledger value using canonical JSON."""

    return dumps(to_jsonable(value), canonical=True, trailing_newline=False)


def decode_json(value: str | None, field: str) -> Any:
    """Decode a JSON column and translate malformed data to StateError."""

    if value is None:
        return None
    try:
        return loads(value)
    except (TypeError, ValueError) as exc:
        raise StateError(f"Malformed {field} in state database") from exc


def decode_row(
    row: sqlite3.Row,
    json_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Convert a SQLite row and decode its explicitly named JSON columns."""

    result = dict(row)
    for column in json_columns:
        result[column.removesuffix("_json")] = decode_json(result.pop(column), column)
    return result


def identity_payload(identity: Any) -> dict[str, Any]:
    """Return the canonical mapping used for identity comparisons."""

    if isinstance(identity, DftCalculationIdentity):
        payload = identity.scientific_payload()
        payload["calculation_id"] = identity.calculation_id
        return payload
    if hasattr(identity, "to_dict"):
        return dict(identity.to_dict())
    if isinstance(identity, Mapping):
        return dict(identity)
    raise TypeError("identity must be a domain identity record or mapping")
