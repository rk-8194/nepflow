"""Focused tests for Materials Project identity, conversion, and injection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure

from nepflow.stages.generation.generators.materials_project.cache import (
    MaterialsProjectCache,
    MaterialsProjectCacheError,
    MaterialsProjectQuery,
)
from nepflow.stages.generation.generators.materials_project.conversion import (
    documents_to_ase,
)
from nepflow.stages.generation.generators.materials_project.fetcher import (
    MaterialsProjectFetcher,
)
from nepflow.stages.generation.generators.materials_project_generator import (
    MaterialsProjectGenerator,
)


def make_document(formula: str = "Si") -> SimpleNamespace:
    symbols = ["Si"] if formula == "Si" else ["Si", "Ge"]
    structure = Structure(
        Lattice.cubic(3.0),
        symbols,
        [[0, 0, 0], [0.5, 0.5, 0.5]][: len(symbols)],
    )
    composition = structure.composition
    return SimpleNamespace(
        material_id=f"mp-{formula}",
        formula_pretty=formula,
        composition=composition,
        structure=structure,
        symmetry=SimpleNamespace(crystal_system="cubic", number=229),
        energy_above_hull=0.1,
    )


def test_query_identity_includes_type_inputs_and_schema() -> None:
    pure = MaterialsProjectQuery(
        query_type="pure_structures",
        elements=("Si",),
        crystal_structures=("bcc",),
    )
    compound = MaterialsProjectQuery(
        query_type="compounds",
        elements=("Si", "Ge"),
        max_per_query=5,
    )

    assert pure.query_id != compound.query_id
    assert "schema_version" not in pure.payload()
    assert (
        pure.query_id
        == MaterialsProjectQuery(
            query_type="pure_structures",
            elements=("Si",),
            crystal_structures=("bcc",),
        ).query_id
    )


def test_cache_rejects_corruption_and_identity_mismatch(tmp_path: Path) -> None:
    cache = MaterialsProjectCache(tmp_path)
    query = MaterialsProjectQuery("compounds", ("Si", "Ge"), max_per_query=2)
    cache.save(query, [{"numbers": [14], "positions": [[0, 0, 0]]}])
    assert cache.load(query) == [{"numbers": [14], "positions": [[0, 0, 0]]}]

    path = cache.path_for(query)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(MaterialsProjectCacheError, match="Malformed"):
        cache.load(query)

    other = MaterialsProjectQuery("compounds", ("Si", "Ge"), max_per_query=3)
    other_path = cache.path_for(other)
    other_path.write_text(
        json.dumps(
            {
                "schema_version": "materials-project-cache-v1",
                "query_id": query.query_id,
                "query": query.payload(),
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MaterialsProjectCacheError, match="identity|query"):
        cache.load(other)


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_cache_requires_matching_record_content_hash(tmp_path: Path, mutation: str) -> None:
    cache = MaterialsProjectCache(tmp_path)
    query = MaterialsProjectQuery("compounds", ("Si",), max_per_query=1)
    cache.save(query, [{"material_id": "mp-1"}])
    path = cache.path_for(query)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["records_sha256"]
    else:
        payload["records_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MaterialsProjectCacheError, match="records"):
        cache.load(query)


def test_cache_save_is_atomic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    cache = MaterialsProjectCache(tmp_path)
    query = MaterialsProjectQuery("pure_structures", ("Si",), ("bcc",))

    cache.save(query, [{"material_id": "mp-1"}])

    assert cache.path_for(query).exists()
    assert list(tmp_path.glob("*.tmp")) == []


class FakeSummary:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.documents)


def test_fetcher_uses_injected_client_and_cached_identity(tmp_path: Path) -> None:
    summary = FakeSummary([make_document()])
    client = SimpleNamespace(materials=SimpleNamespace(summary=summary))
    fetcher = MaterialsProjectFetcher(client=client, cache_dir=tmp_path)

    first = fetcher.fetch_structures(["Si"], ["bcc"])
    second = fetcher.fetch_structures(["Si"], ["bcc"])

    assert len(first) == len(second) == 1
    assert len(summary.calls) == 1


def test_mp_conversion_and_generator_separate_requested_from_actual_composition() -> None:
    atoms = documents_to_ase([make_document("SiGe")])[0]
    assert atoms.info["actual_composition"] == {"Ge": 0.5, "Si": 0.5}

    class FakeFetcher:
        def fetch_compounds(self, elements, max_per_query, use_cache):
            del elements, max_per_query, use_cache
            return [atoms.copy()]

    requested = {"Si": 0.25, "Ge": 0.75}
    result = MaterialsProjectGenerator(FakeFetcher(), max_per_composition=1).generate(
        requested, ["bcc"]
    )[0]

    assert result.info["composition"] == requested
    assert result.info["actual_composition"] == {"Ge": 0.5, "Si": 0.5}
