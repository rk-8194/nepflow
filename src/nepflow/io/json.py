"""Strict UTF-8 JSON serialization backed by atomic writes."""

from __future__ import annotations

import copy
import json as _json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from nepflow.errors import StateError

from .atomic import atomic_write_text

_MISSING = object()
_ErrorT = TypeVar("_ErrorT", bound=Exception)


def _reject_json_constant(value: str) -> Any:
    """Reject Python-only numeric constants that are not valid JSON."""

    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def to_jsonable(value: Any) -> Any:
    """Convert common repository values to strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def dumps(
    value: Any,
    *,
    canonical: bool = False,
    indent: int | None = 2,
    trailing_newline: bool | None = None,
) -> str:
    """Serialize JSON deterministically, rejecting NaN and infinity.

    Canonical output is compact and newline-free because it is used as content
    identity.  Persistent human-readable output is sorted and newline-terminated
    by default.
    """

    if trailing_newline is None:
        trailing_newline = not canonical
    serialized = _json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":") if canonical else None,
        ensure_ascii=True if canonical else False,
        allow_nan=False,
        indent=None if canonical else indent,
    )
    return serialized + ("\n" if trailing_newline else "")


def loads(text: str) -> Any:
    """Deserialize JSON text without converting malformed input to a default."""

    return _json.loads(text, parse_constant=_reject_json_constant)


def read_json(
    path: str | Path,
    *,
    error_type: type[_ErrorT] = StateError,
    missing_error_type: type[Exception] | None = None,
    default: Any = _MISSING,
    require_object: bool = False,
) -> Any:
    """Read one strict UTF-8 JSON document.

    Missing files are raised as ``FileNotFoundError`` by default so callers can
    make an explicit optional-file decision before calling this function.
    Malformed, unreadable, or non-UTF-8 content is converted to the caller's
    declared typed error.
    """

    source = Path(path)
    try:
        value = loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if default is not _MISSING:
            value = copy.deepcopy(default)
        elif missing_error_type is None:
            raise FileNotFoundError(f"JSON file not found: {source}") from exc
        else:
            raise missing_error_type(f"JSON file not found: {source}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise error_type(f"Could not read JSON file {source}: {exc}") from exc
    if require_object and not isinstance(value, dict):
        raise error_type(f"JSON file must contain an object: {source}")
    return value


def read_json_object(
    path: str | Path,
    *,
    error_type: type[_ErrorT] = StateError,
    missing_error_type: type[Exception] | None = None,
    default: Any = _MISSING,
) -> dict[str, Any]:
    """Read strict JSON and require an object at the persistence boundary."""
    return read_json(
        path,
        error_type=error_type,
        missing_error_type=missing_error_type,
        default=default,
        require_object=True,
    )


def write_json(
    path: str | Path,
    value: Any,
    *,
    canonical: bool = False,
    indent: int | None = 2,
    durable: bool = True,
) -> None:
    """Serialize and atomically write one JSON document."""

    atomic_write_text(
        path,
        dumps(value, canonical=canonical, indent=indent),
        encoding="utf-8",
        durable=durable,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for content identity."""

    return dumps(value, canonical=True, trailing_newline=False).encode("utf-8")


__all__ = [
    "canonical_json_bytes",
    "dumps",
    "loads",
    "read_json",
    "read_json_object",
    "to_jsonable",
    "write_json",
]
