"""Materials Project query orchestration and source provenance."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ase import Atoms

from .cache import MaterialsProjectCache, MaterialsProjectCacheError, MaterialsProjectQuery
from .client import MaterialsProjectClient
from .conversion import (
    atoms_from_cache_records,
    atoms_to_cache_records,
    documents_to_ase,
    filter_pure_documents,
    pure_records_to_ase,
    serialize_structure,
)

logger = logging.getLogger(__name__)


class MaterialsProjectFetcher:
    """Fetch and cache MP structures through injectable boundaries."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        request_timeout: float = 60.0,
        *,
        client: Any | None = None,
        client_factory: Any | None = None,
        cache: MaterialsProjectCache | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("MP_API_KEY")
        if not resolved_key and client is None:
            raise ValueError(
                "API key required. Set MP_API_KEY environment variable or pass api_key parameter."
            )
        self.api_key = resolved_key or "injected-client"
        resolved_cache_dir = cache_dir or Path.home() / ".cache" / "nepflow" / "mp"
        self.cache = cache or MaterialsProjectCache(resolved_cache_dir)
        self.cache_dir = self.cache.directory
        self.request_timeout = request_timeout
        self._client_adapter = MaterialsProjectClient(
            self.api_key,
            request_timeout,
            client=client,
            client_factory=client_factory,
        )

    @property
    def client(self) -> Any:
        return self._client_adapter.client

    def _cache_path(self, query_id: str) -> Path:
        return self.cache.directory / f"{query_id}.json"

    def _calculate_query_id(
        self,
        elements: Sequence[str],
        structures: Sequence[str] = (),
        *,
        query_type: str = "pure_structures",
        max_per_query: int | None = None,
    ) -> str:
        return MaterialsProjectQuery(
            query_type=query_type,
            elements=tuple(elements),
            crystal_structures=tuple(structures),
            max_per_query=max_per_query,
        ).query_id

    def fetch_structures(
        self,
        elements: Sequence[str],
        crystal_structures: Sequence[str],
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        for element in elements:
            query = MaterialsProjectQuery(
                query_type="pure_structures",
                elements=(element,),
                crystal_structures=tuple(crystal_structures),
            )
            records = self.cache.load(query) if use_cache else None
            if records is None:
                try:
                    documents = self.client.materials.summary.search(
                        elements=[element],
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "composition",
                            "structure",
                            "symmetry",
                        ],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Materials Project query failed for "
                        f"element={element}, structures={list(crystal_structures)}"
                    ) from exc
                records = [
                    serialize_structure(document)
                    for document in filter_pure_documents(documents or [], crystal_structures)
                ]
                for record in records:
                    record["query_id"] = query.query_id
                self.cache.save(query, records)
            else:
                for record in records:
                    record.setdefault("query_id", query.query_id)
            all_results.extend(records)
        return all_results

    def get_lattice_parameters(
        self,
        element: str,
        structure: str,
        use_cache: bool = True,
    ) -> dict[str, Any] | None:
        results = self.fetch_structures([element], [structure], use_cache=use_cache)
        return results[0] if results else None

    def fetch_compounds(
        self,
        elements: Sequence[str],
        max_per_query: int = 50,
        use_cache: bool = True,
    ) -> list[Atoms]:
        query = MaterialsProjectQuery(
            query_type="compounds",
            elements=tuple(elements),
            max_per_query=max_per_query,
        )
        records = self.cache.load(query) if use_cache else None
        if records is not None:
            try:
                return atoms_from_cache_records(records)
            except (KeyError, TypeError, ValueError) as exc:
                raise MaterialsProjectCacheError(
                    f"Malformed Materials Project ASE cache for {query.query_id}"
                ) from exc

        try:
            documents = self.client.materials.summary.search(
                elements=list(elements),
                num_elements=(1, len(elements)),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "composition",
                    "structure",
                    "symmetry",
                    "energy_above_hull",
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Materials Project query failed for elements={list(elements)}"
            ) from exc

        element_set = set(elements)
        filtered = [
            document
            for document in (documents or [])
            if set(document.composition.as_dict().keys()).issubset(element_set)
        ]
        filtered.sort(key=lambda document: getattr(document, "energy_above_hull", 0) or 0)
        atoms_list = documents_to_ase(filtered[:max_per_query])
        for atoms in atoms_list:
            atoms.info["query_id"] = query.query_id
        self.cache.save(query, atoms_to_cache_records(atoms_list))
        return atoms_list

    def fetch_pure_element_structures(
        self,
        elements: Sequence[str],
        crystal_structures: Sequence[str],
        use_cache: bool = True,
    ) -> list[Atoms]:
        records = self.fetch_structures(elements, crystal_structures, use_cache=use_cache)
        return pure_records_to_ase(records)

    def clear_cache(self) -> None:
        self.cache.clear()


def build_materials_project_fetcher(
    api_key: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    request_timeout: float = 60.0,
    client: Any | None = None,
    client_factory: Any | None = None,
) -> MaterialsProjectFetcher:
    """Build the MP adapter from typed composition-root inputs."""

    return MaterialsProjectFetcher(
        api_key=api_key,
        cache_dir=cache_dir,
        request_timeout=request_timeout,
        client=client,
        client_factory=client_factory,
    )
