"""Behavior locks for the focused configurational generator owners."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.stages.generation.generators.composition_primitives import (
    allocate_crystal_quota,
)
from nepflow.stages.generation.generators.random_solution import (
    RandomSolidSolutionGenerator,
)
from nepflow.stages.generation.generators.segregated import SegregatedGenerator
from nepflow.stages.generation.generators.sqs import SQSGenerationError, SQSGenerator


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

    def generate(self, *, primitive, max_size, target_concentrations, random_seed):
        del primitive, max_size, target_concentrations
        self.seeds.append(random_seed)
        return Atoms("Si4", cell=[3, 3, 3], pbc=True)


def test_sqs_uses_declared_backend_and_records_effective_seed() -> None:
    backend = FakeSQSBackend()
    results = SQSGenerator(
        n_structures=2,
        random_seed=11,
        backend=backend,
    ).generate({"Si": 0.5, "Ge": 0.5}, ["bcc"], target_n_atoms=8)

    assert backend.seeds == [11, 12]
    assert [item.info["random_seed"] for item in results] == [11, 12]


def test_sqs_backend_failure_is_explicit_and_never_falls_back() -> None:
    class FailingBackend:
        def generate(self, **kwargs):
            del kwargs
            raise RuntimeError("backend unavailable")

    with pytest.raises(SQSGenerationError, match="backend unavailable"):
        SQSGenerator(backend=FailingBackend()).generate(
            {"Si": 0.5, "Ge": 0.5}, ["bcc"], target_n_atoms=8
        )
