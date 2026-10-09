"""Focused contracts for typed perturbation tasks and shared ownership."""

import pickle
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.domain.identities import calculate_structure_id
from nepflow.stages.generation.perturbations.models import (
    PerturbationCounts,
    PerturbationSettings,
    PerturbationTask,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.supercell import build_target_supercell


def test_worker_task_is_pickle_serializable_and_carries_exact_identity() -> None:
    base = Atoms("Si2", cell=np.eye(3) * 3.0, pbc=True)
    task = PerturbationTask(
        base=base,
        base_structure_id=calculate_structure_id(base),
        settings=PerturbationSettings(random_seed=7),
        counts=PerturbationCounts(n_rattled=2),
        seed=101,
    )

    restored = pickle.loads(pickle.dumps(task))

    assert restored.base_structure_id == task.base_structure_id
    assert restored.settings.random_seed == 7
    assert restored.counts.n_rattled == 2
    assert restored.seed == 101
    assert restored.base.get_chemical_symbols() == ["Si", "Si"]
    assert not hasattr(restored, "progress_queue")


def test_bounded_worker_task_has_no_progress_event_transport() -> None:
    base = Atoms("Si2", cell=np.eye(3) * 3.0, pbc=True)
    prepared = base.copy()
    task = PerturbationTask(
        base=base,
        base_structure_id=calculate_structure_id(base),
        settings=PerturbationSettings(random_seed=7),
        counts=PerturbationCounts(n_vacancies=10),
        seed=101,
        family="vacancy",
        slot_start=8,
        slot_stop=10,
        prepared_supercell=prepared,
    )

    restored = pickle.loads(pickle.dumps(task))

    assert restored.prepared_supercell is not None
    assert restored.family == "vacancy"
    assert (restored.slot_start, restored.slot_stop) == (8, 10)
    assert not hasattr(restored, "progress_queue")

    import nepflow.stages.generation.perturbations.models as models_module

    assert not hasattr(models_module, "PerturbationProgressEvent")


def test_target_supercell_preserves_species_cell_and_pbc() -> None:
    base = Atoms(
        "SiGe",
        positions=[[0, 0, 0], [1, 1, 1]],
        cell=np.eye(3) * 2.0,
        pbc=[True, False, True],
    )

    expanded = build_target_supercell(base, target_n_atoms=8)

    assert expanded is not None
    assert len(expanded) == 16
    assert expanded.pbc.tolist() == [True, False, True]
    assert Counter(expanded.get_chemical_symbols()) == Counter({"Si": 8, "Ge": 8})
    np.testing.assert_allclose(expanded.cell.array, np.eye(3) * 4.0)


def test_provenance_record_is_canonical_and_retains_legacy_metadata() -> None:
    base = Atoms("Si2", cell=np.eye(3) * 3.0, pbc=True)
    base.info.update(
        {
            "seed_id": "seed_000001",
            "source": "fixture",
            "composition": {"Si": 1.0},
            "crystal_structure": "fcc",
            "configurational_type": "test_base",
        }
    )
    candidate = base.copy()

    record = annotate_generation_provenance(
        candidate,
        base,
        "vacancy",
        random_seed=13,
        parameters={"n_vacancies": 1},
    )

    assert record.provenance.perturbation_family == "vacancy"
    assert record.provenance.parent_structure_id == calculate_structure_id(base)
    assert candidate.info["perturbation_type"] == "vacancy"
    assert candidate.info["random_seed"] == 13
    assert candidate.info["generation_provenance"] == record.provenance.to_dict()
