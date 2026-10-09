"""Composition-aware deterministic supercell contracts."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.stages.generation.generators.random_solution import (
    RandomSolidSolutionGenerator,
)
from nepflow.stages.generation.generators.segregated import SegregatedGenerator
from nepflow.stages.generation.generators.sqs import SQSGenerator
from nepflow.stages.generation.supercell import (
    CompositionRealizabilityError,
    build_target_supercell,
)


def parent(cell: tuple[float, float, float], symbols: str = "W") -> Atoms:
    return Atoms(
        symbols,
        cell=np.diag(cell),
        positions=np.zeros((len(Atoms(symbols)), 3)),
        pbc=True,
    )


def test_cubic_parent_uses_balanced_diagonal_repeat() -> None:
    result = build_target_supercell(parent((2.0, 2.0, 2.0)), target_n_atoms=8)

    assert result is not None
    assert len(result) == 8
    assert result.info["supercell_repeat"] == (2, 2, 2)


def test_anisotropic_parent_search_is_geometry_aware_and_deterministic() -> None:
    first = build_target_supercell(parent((2.0, 4.0, 6.0)), target_n_atoms=8)
    second = build_target_supercell(parent((2.0, 4.0, 6.0)), target_n_atoms=8)

    assert first is not None and second is not None
    assert first.info["supercell_repeat"] == second.info["supercell_repeat"]
    lengths = np.linalg.norm(first.cell.array, axis=1)
    assert max(lengths) / min(lengths) < 2.0


def test_repeat_tie_breaking_is_lexically_stable() -> None:
    first = build_target_supercell(parent((2.0, 2.0, 2.0)), target_n_atoms=4)
    second = build_target_supercell(parent((2.0, 2.0, 2.0)), target_n_atoms=4)

    assert first is not None and second is not None
    assert first.info["supercell_repeat"] == second.info["supercell_repeat"]


def test_already_large_parent_is_returned_unchanged() -> None:
    source = parent((2.0, 2.0, 2.0), "W4")

    result = build_target_supercell(source, target_n_atoms=2)

    assert result is not None
    assert result is not source
    assert len(result) == len(source)
    np.testing.assert_allclose(result.cell.array, source.cell.array)
    assert result.get_chemical_symbols() == source.get_chemical_symbols()


def test_binary_composition_is_recorded_as_integer_counts_and_fractions() -> None:
    result = build_target_supercell(
        "W",
        "bcc",
        3,
        composition={"W": 0.5, "Cr": 0.5},
        composition_tolerance=0.0,
    )

    assert result is not None
    assert len(result) in {2, 4}
    assert abs(len(result) - 3) == 1
    expected_count = len(result) // 2
    assert result.info["composition_target_counts"] == {
        "Cr": expected_count,
        "W": expected_count,
    }
    assert result.info["composition_target_fractions"] == {"Cr": 0.5, "W": 0.5}
    assert result.info["composition_target_error"] == 0.0


def test_ternary_composition_search_finds_nearby_valid_size() -> None:
    result = build_target_supercell(
        "W",
        "fcc",
        8,
        composition={"W": 1.0 / 3.0, "Cr": 1.0 / 3.0, "Y": 1.0 / 3.0},
        composition_tolerance=0.0,
    )

    assert result is not None
    assert len(result) == 9
    assert result.info["composition_target_counts"] == {"Cr": 3, "W": 3, "Y": 3}


def test_no_allowed_repeat_fails_explicitly() -> None:
    with pytest.raises(CompositionRealizabilityError, match="no allowed diagonal repeat"):
        build_target_supercell(
            "W",
            "fcc",
            8,
            composition={"W": 0.34, "Cr": 0.33, "Y": 0.33},
            composition_tolerance=0.0,
            max_n_atoms=8,
        )


class BalancedSQSBackend:
    def generate(self, *, primitive, supercells, target_concentrations, random_seed):
        del primitive, random_seed
        result = supercells[0].copy()
        result.set_chemical_symbols(["W"] * 4 + ["Cr"] * 4)
        return result


def test_random_sqs_and_segregated_preserve_realization_metadata() -> None:
    composition = {"W": 0.5, "Cr": 0.5}
    random = RandomSolidSolutionGenerator(
        n_structures=1,
        composition_tolerance=0.0,
    ).generate(composition, ["bcc"], target_n_atoms=8)[0]
    segregated = SegregatedGenerator(
        n_structures=1,
        composition_tolerance=0.0,
    ).generate(composition, ["bcc"], target_n_atoms=8)[0]
    sqs = SQSGenerator(
        n_structures=1,
        backend=BalancedSQSBackend(),
        composition_tolerance=0.0,
    ).generate(composition, ["bcc"], target_n_atoms=8)[0]

    for atoms in (random, segregated, sqs):
        counts = Counter(atoms.get_chemical_symbols())
        assert atoms.info["requested_composition"] == composition
        assert atoms.info["composition_counts"] == dict(sorted(counts.items()))
        assert atoms.info["actual_composition"] == {"Cr": 0.5, "W": 0.5}
        assert atoms.info["composition_error"] == 0.0
        assert counts["Cr"] == counts["W"]

    assert len(random) == 8
    assert len(segregated) == 8
    assert len(sqs) == 8
