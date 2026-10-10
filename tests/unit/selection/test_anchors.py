"""Exact physical-identity and mandatory-anchor selection contracts."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms
from ase.io import read, write

from nepflow.stages.selection import strategy

SEED_LINEAGE_FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "structures" / "seed_lineage.extxyz.fixture"
)


def _seed_project(tmp_path):
    lineage = read(SEED_LINEAGE_FIXTURE, index=":", format="extxyz")
    base = next(
        atoms for atoms in lineage if str(atoms.info.get("perturbation_type", "")) == "unperturbed"
    )
    seed_path = tmp_path / "structures" / "seeds" / "base_structures.xyz"
    seed_path.parent.mkdir(parents=True)
    write(seed_path, base, format="extxyz")
    return lineage, base


def _fingerprint(atoms):
    return (
        tuple(atoms.get_chemical_symbols()),
        tuple(np.asarray(atoms.cell.array).round(12).ravel()),
        tuple(np.asarray(atoms.positions).round(12).ravel()),
        tuple(bool(value) for value in atoms.pbc),
    )


def test_seed_anchor_uses_physical_identity_across_candidate_orderings(tmp_path):
    lineage, base = _seed_project(tmp_path)
    expected = _fingerprint(base)

    for ordered in permutations(lineage):
        indices = strategy.resolve_seed_indices(tmp_path, list(ordered))
        assert len(indices) == 1
        assert _fingerprint(list(ordered)[indices[0]]) == expected


def test_duplicate_anchor_categories_deduplicate_one_candidate(tmp_path):
    lineage, _ = _seed_project(tmp_path)
    settings = {
        "target_train": 2,
        "composition_aware_fps": False,
        "mean_descriptor": True,
        "tolerance": 1,
        "max_iterations": 4,
    }
    structures = [object() for _ in lineage]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            strategy,
            "select_farthest_points_for_target",
            lambda *args, **kwargs: ([0], 0.25),
        )
        seed = strategy.resolve_seed_indices(tmp_path, lineage)
        selected, _ = strategy.select_training_set(
            np.arange(len(lineage), dtype=float).reshape(len(lineage), 1),
            structures,
            settings,
            algorithm_id="fps",
            seed_indices=seed,
            single_element_elastic_indices=seed,
            elastic_indices=seed,
        )

    assert selected.count(seed[0]) == 1


def test_unresolved_mandatory_seed_anchor_fails_explicitly(tmp_path):
    lineage, _ = _seed_project(tmp_path)
    with pytest.raises(ValueError, match="no physically identical"):
        strategy.resolve_seed_indices(tmp_path, lineage[1:])


def test_elastic_anchor_categories_have_explicit_unary_boundary():
    unary = Atoms("Si2", positions=np.zeros((2, 3)))
    unary.info["perturbation_type"] = "elastic_stress"
    unary.info["composition"] = {"Si": 1.0}
    binary = Atoms("SiGe", positions=np.zeros((2, 3)))
    binary.info["perturbation_type"] = "elastic_stress"

    assert strategy.find_elastic_stress_indices([unary, binary]) == [0, 1]
    assert strategy.find_single_element_elastic_stress_indices([unary, binary]) == [0]
