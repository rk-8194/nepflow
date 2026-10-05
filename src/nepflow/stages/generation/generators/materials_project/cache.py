"""Identity-safe Materials Project JSON cache storage."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nepflow.errors import ArtifactError
from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import read_json_object, to_jsonable, write_json

MATERIALS_PROJECT_CACHE_SCHEMA = "materials-project-cache-v2"


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
            "query_type": self.query_type,
            "elements": sorted(set(self.elements)),
            "crystal_structures": sorted(set(self.crystal_structures)),
            "max_per_query": self.max_per_query,
        }

    @property
    def query_id(self) -> str:
        return sha256_canonical_json(
            {
                "schema_version": MATERIALS_PROJECT_CACHE_SCHEMA,
                "query": self.payload(),
            }
        )


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
            payload = read_json_object(path, error_type=MaterialsProjectCacheError)
        except (OSError, UnicodeError, ValueError, MaterialsProjectCacheError) as exc:
            raise MaterialsProjectCacheError(f"Malformed Materials Project cache: {path}") from exc
        if payload.get("query_id") != query.query_id:
            raise MaterialsProjectCacheError(f"Materials Project cache identity mismatch: {path}")
        if payload.get("query") != query.payload():
            raise MaterialsProjectCacheError(f"Materials Project cache query mismatch: {path}")
        if payload.get("schema_version") != MATERIALS_PROJECT_CACHE_SCHEMA:
            raise MaterialsProjectCacheError(f"Unsupported Materials Project cache schema: {path}")
        records = payload.get("records")
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise MaterialsProjectCacheError(f"Invalid Materials Project cache records: {path}")
        expected_records_hash = payload.get("records_sha256")
        if not isinstance(expected_records_hash, str) or not expected_records_hash:
            raise MaterialsProjectCacheError(
                f"Materials Project cache has no records content identity: {path}"
            )
        if sha256_canonical_json(records) != expected_records_hash:
            raise MaterialsProjectCacheError(
                f"Materials Project cache records content mismatch: {path}"
            )
        return [dict(record) for record in records]

    def save(self, query: MaterialsProjectQuery, records: Sequence[Mapping[str, Any]]) -> Path:
        path = self.path_for(query)
        normalized_records = [dict(to_jsonable(record)) for record in records]
        envelope = {
            "schema_version": MATERIALS_PROJECT_CACHE_SCHEMA,
            "query_id": query.query_id,
            "query": query.payload(),
            "records": normalized_records,
            "records_sha256": sha256_canonical_json(normalized_records),
        }
        write_json(path, envelope)
        return path

    def clear(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
