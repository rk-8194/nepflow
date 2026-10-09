"""Behavior locks for the focused configurational generator owners."""

import inspect
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.build import bulk

from nepflow.stages.generation.generators.composition_primitives import (
    allocate_crystal_quota,
    calculate_composition_realization,
)
from nepflow.stages.generation.generators.random_solution import (
    RandomSolidSolutionGenerator,
)
from nepflow.stages.generation.generators.segregated import SegregatedGenerator
from nepflow.stages.generation.generators.sqs import (
    IcetSQSBackend,
    SQSGenerationError,
    SQSGenerator,
)
from nepflow.stages.generation.supercell import build_target_supercell


def test_quota_is_exact_and_preserves_configured_order() -> None:
    assert allocate_crystal_quota(5, ["bcc", "fcc", "hcp"]) == [
        ("bcc", 2),
        ("fcc", 2),
        ("hcp", 1),
    ]


def test_random_and_segregated_generators_preserve_per_crystal_quota() -> None:
    composition = {"Si": 0.5, "Ge": 0.5}
    structures = ["bcc", "fcc"]

    random_results = RandomSolidSolutionGenerator(
        n_structures=4,
        rng=np.random.default_rng(7),
    ).generate(composition, structures, target_n_atoms=8)
    segregated_results = SegregatedGenerator(
        n_structures=4,
        rng=np.random.default_rng(7),
    ).generate(composition, structures, target_n_atoms=8)

    assert Counter(item.info["crystal_structure"] for item in random_results) == {
        "bcc": 2,
        "fcc": 2,
    }
    assert Counter(item.info["crystal_structure"] for item in segregated_results) == {
        "bcc": 2,
        "fcc": 2,
    }
    assert all("actual_composition" in item.info for item in random_results)
    assert all("actual_composition" in item.info for item in segregated_results)


class FakeSQSBackend:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.supercells: list[tuple[Atoms, ...]] = []

    def generate(self, *, primitive, supercells, target_concentrations, random_seed):
        del primitive
        self.seeds.append(random_seed)
        self.supercells.append(tuple(supercells))
        result = supercells[0].copy()
        realization = calculate_composition_realization(target_concentrations, len(result))
        symbols = [
            element
            for element, count in realization.counts.items()
            for _ in range(count)
        ]
        result.set_chemical_symbols(symbols)
        return result


def test_sqs_uses_declared_backend_and_records_effective_seed() -> None:
    backend = FakeSQSBackend()
    results = SQSGenerator(
        n_structures=2,
        random_seed=11,
        backend=backend,
    ).generate({"Si": 0.5, "Ge": 0.5}, ["bcc"], target_n_atoms=8)

    assert backend.seeds == [11, 12]
    assert [item.info["random_seed"] for item in results] == [11, 12]
    assert all(
        len(supercells) == 1 and len(supercells[0]) == 8
        for supercells in backend.supercells
    )


def test_sqs_backend_failure_is_explicit_and_never_falls_back() -> None:
    class FailingBackend:
        def generate(self, *, primitive, supercells, target_concentrations, random_seed):
            del primitive, supercells, target_concentrations, random_seed
            raise RuntimeError("backend unavailable")

    with pytest.raises(SQSGenerationError, match="backend unavailable"):
        SQSGenerator(backend=FailingBackend()).generate(
            {"Si": 0.5, "Ge": 0.5}, ["bcc"], target_n_atoms=8
        )


@pytest.mark.parametrize("crystal_structure", ("bcc", "fcc", "hcp"))
def test_icet_adapter_matches_the_installed_explicit_supercell_signature(
    crystal_structure: str,
) -> None:
    pytest.importorskip("icet")
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    signature = inspect.signature(generate_sqs_from_supercells)
    assert "supercells" in signature.parameters
    assert "max_size" not in signature.parameters

    primitive = (
        bulk("Cr", "hcp", a=3.0, c=3.0 * 1.633)
        if crystal_structure == "hcp"
        else bulk("Cr", crystal_structure, a=3.0)
    )
    supercell = build_target_supercell(
        primitive,
        8,
        composition={"Cr": 0.875, "W": 0.125},
        composition_tolerance=0.0,
        raise_on_error=True,
    )
    calls: list[dict[str, object]] = []

    def checked_generator(
        cluster_space,
        supercells,
        target_concentrations,
        T_start=5.0,
        T_stop=0.001,
        n_steps=None,
        optimality_weight=1.0,
        random_seed=None,
        random_start=True,
        tol=1.0e-5,
    ):
        values = {
            "cluster_space": cluster_space,
            "supercells": supercells,
            "target_concentrations": target_concentrations,
            "T_start": T_start,
            "T_stop": T_stop,
            "n_steps": n_steps,
            "optimality_weight": optimality_weight,
            "random_seed": random_seed,
            "random_start": random_start,
            "tol": tol,
        }
        signature.bind(**values)
        calls.append(values)
        result = supercells[0].copy()
        result.set_chemical_symbols(["Cr"] * 7 + ["W"])
        return result

    result = IcetSQSBackend(ClusterSpace, checked_generator, n_steps=10).generate(
        primitive=primitive,
        supercells=(supercell,),
        target_concentrations={"Cr": 0.875, "W": 0.125},
        random_seed=19,
    )

    assert result is not supercell
    assert len(calls) == 1
    assert calls[0]["supercells"] == [supercell]
    assert "max_size" not in calls[0]


def test_icet_backend_generates_a_planned_binary_bcc_supercell() -> None:
    pytest.importorskip("icet")
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    composition = {"Cr": 0.875, "W": 0.125}
    primitive = bulk("Cr", "bcc", a=3.0)
    supercell = build_target_supercell(
        primitive,
        8,
        composition=composition,
        composition_tolerance=0.0,
        raise_on_error=True,
    )
    backend = IcetSQSBackend(ClusterSpace, generate_sqs_from_supercells, n_steps=25)
    first = backend.generate(
        primitive=primitive,
        supercells=(supercell,),
        target_concentrations=composition,
        random_seed=23,
    )
    second = backend.generate(
        primitive=primitive,
        supercells=(supercell,),
        target_concentrations=composition,
        random_seed=23,
    )

    assert len(first) == len(supercell) == 8
    assert Counter(first.get_chemical_symbols()) == Counter({"Cr": 7, "W": 1})
    np.testing.assert_allclose(first.cell.array, supercell.cell.array)
    assert first.get_chemical_symbols() == second.get_chemical_symbols()
