"""Canonical SHA-256 helpers for content and artifact identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nepflow.errors import ArtifactError

from .json import canonical_json_bytes as _canonical_json_bytes


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded text."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_file(
    path: str | Path,
    *,
    required: bool = True,
    error_type: type[Exception] = ArtifactError,
    chunk_size: int = 1024 * 1024,
) -> str | None:
    """Hash a file, explicitly distinguishing required and optional artifacts."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        if not required:
            return None
        raise error_type(f"Cannot hash required artifact {source}: {exc}") from exc
    except OSError as exc:
        raise error_type(f"Cannot hash required artifact {source}: {exc}") from exc
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical JSON bytes."""

    return sha256_bytes(_canonical_json_bytes(value))


__all__ = [
    "sha256_bytes",
    "sha256_canonical_json",
    "sha256_file",
    "sha256_text",
]
