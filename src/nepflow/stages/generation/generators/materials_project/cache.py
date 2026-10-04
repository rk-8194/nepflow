"""Identity-safe Materials Project JSON cache storage."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nepflow.errors import ArtifactError
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import dumps, loads

MATERIALS_PROJECT_CACHE_SCHEMA = "materials-project-cache-v1"


class MaterialsProjectCacheError(ArtifactError):
    """Raised for malformed, stale, or identity-mismatched MP cache data."""


@dataclass(frozen=True, slots=True)
class MaterialsProjectQuery:
    """All query-affecting inputs used to identify one cache entry."""

    query_type: str
    elements: tuple[str, ...]
    crystal_structures: tuple[str, ...] = ()
    max_per_query: int | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "cache_schema": MATERIALS_PROJECT_CACHE_SCHEMA,
            "query_type": self.query_type,
            "elements": sorted(set(self.elements)),
            "crystal_structures": sorted(set(self.crystal_structures)),
            "max_per_query": self.max_per_query,
        }

    @property
    def query_id(self) -> str:
        return sha256_canonical_json(self.payload())


class MaterialsProjectCache:
    """Read/write cache envelopes with exact identity verification."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, query: MaterialsProjectQuery) -> Path:
        return self.directory / f"{query.query_id}.json"

    def load(self, query: MaterialsProjectQuery) -> list[dict[str, Any]] | None:
        path = self.path_for(query)
        if not path.exists():
            return None
        try:
            payload = loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise MaterialsProjectCacheError(f"Malformed Materials Project cache: {path}") from exc
        if not isinstance(payload, Mapping):
            raise MaterialsProjectCacheError(f"Invalid Materials Project cache envelope: {path}")
        if payload.get("cache_schema") != MATERIALS_PROJECT_CACHE_SCHEMA:
            raise MaterialsProjectCacheError(f"Unsupported Materials Project cache schema: {path}")
        if payload.get("query_id") != query.query_id:
            raise MaterialsProjectCacheError(f"Materials Project cache identity mismatch: {path}")
        if payload.get("query") != query.payload():
            raise MaterialsProjectCacheError(f"Materials Project cache query mismatch: {path}")
        records = payload.get("records")
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise MaterialsProjectCacheError(f"Invalid Materials Project cache records: {path}")
        return [dict(record) for record in records]

    def save(self, query: MaterialsProjectQuery, records: Sequence[Mapping[str, Any]]) -> Path:
        path = self.path_for(query)
        envelope = {
            "cache_schema": MATERIALS_PROJECT_CACHE_SCHEMA,
            "query_id": query.query_id,
            "query": query.payload(),
            "records": [dict(record) for record in records],
        }
        atomic_write_text(path, dumps(envelope, indent=2))
        return path

    def clear(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
