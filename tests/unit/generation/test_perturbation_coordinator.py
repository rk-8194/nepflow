"""Coordinator execution, defect placement, and worker failure contracts."""

import pickle
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("hiphive")
from ase import Atoms

from nepflow.domain.identities import calculate_structure_id
from nepflow.stages.generation.perturbations.coordinator import (
    PerturbationCoordinator,
    PerturbationTaskError,
    execute_perturbation_task,
)
from nepflow.stages.generation.perturbations.defects import (
    gas_in_vacancy,
    gas_interstitials,
    interstitials,
    vacancies,
    vacancy_interstitial,
)
from nepflow.stages.generation.perturbations.models import (
    PerturbationCounts,
    PerturbationSettings,
    PerturbationTask,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)


def base_atoms() -> Atoms:
    atoms = Atoms(
        "Si4",
        positions=[[2, 2, 2], [6, 2, 2], [2, 6, 2], [2, 2, 6]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    atoms.info.update(
        {
            "seed_id": "seed_000001",
            "source": "coordinator-fixture",
            "elements": ["Si"],
            "configurational_type": "test_base",
        }
    )
    return atoms


def test_defect_families_use_shared_site_placement_and_seed_provenance() -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=17,
        vacancy_range=(0.25, 0.25),
        interstitial_range=(0.25, 0.25),
        interstitial_d_min=1.0,
        gas_elements=("H",),
        gas_interstitial_d_min=0.8,
        max_gas_occupancy=2,
    )

    def run(family):
        return family(
            base,
            base,
            1,
            settings,
            np.random.RandomState(17),
            annotate_generation_provenance,
            seed=17,
        )

    outputs = [
        run(vacancies),
        run(interstitials),
        run(gas_interstitials),
        run(vacancy_interstitial),
        run(gas_in_vacancy),
    ]

    assert all(len(result) == 1 for result in outputs)
    assert [result[0].info["random_seed"] for result in outputs] == [17] * 5
    assert outputs[0][0].info["n_vacancies"] == 1
    assert outputs[1][0].info["n_interstitials"] >= 0
    assert outputs[2][0].info["n_gas_interstitials"] >= 0
    assert outputs[3][0].info["n_vacancies"] == 1
    assert outputs[4][0].info["n_gas_atoms"] >= 0


def test_serial_and_parallel_candidates_are_ordered_and_scientifically_equal() -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    counts = PerturbationCounts(
        n_rattled=0,
        n_vacancies=1,
        n_interstitials=0,
    )

    serial = PerturbationCoordinator(settings=settings).generate_candidates(
        [base], counts=counts, n_workers=1
    )
    parallel = PerturbationCoordinator(settings=settings).generate_candidates(
        [base], counts=counts, n_workers=2
    )

    assert len(serial) == len(parallel)
    for left, right in zip(serial, parallel):
        np.testing.assert_array_equal(left.numbers, right.numbers)
        np.testing.assert_allclose(left.positions, right.positions)
        np.testing.assert_allclose(left.cell.array, right.cell.array)
        assert left.info == right.info

    task = PerturbationCoordinator(settings=settings)._tasks([base], counts)[0]
    result = execute_perturbation_task(task)
    assert len(result.provenance_records) == len(result.candidates)


def test_worker_result_is_pickleable_with_canonical_provenance() -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    counts = PerturbationCounts(
        n_rattled=0,
        n_vacancies=1,
        n_interstitials=0,
    )
    task = PerturbationCoordinator(settings=settings)._tasks([base], counts)[0]

    result = execute_perturbation_task(task)
    restored = pickle.loads(pickle.dumps(result))

    assert restored.task.base_structure_id == result.task.base_structure_id
    assert calculate_structure_id(restored.task.base) == calculate_structure_id(result.task.base)
    assert restored.task.settings == result.task.settings
    assert restored.task.counts == result.task.counts
    assert restored.task.seed == result.task.seed
    assert [calculate_structure_id(item) for item in restored.candidates] == [
        calculate_structure_id(item) for item in result.candidates
    ]
    assert [item.info.get("random_seed") for item in restored.candidates] == [
        item.info.get("random_seed") for item in result.candidates
    ]
    assert [record.to_dict() for record in restored.provenance_records] == [
        record.to_dict() for record in result.provenance_records
    ]
    assert [record.structure_id for record in restored.provenance_records] == [
        record.structure_id for record in result.provenance_records
    ]
    assert all(
        isinstance(record.provenance.realised_composition, MappingProxyType)
        for record in restored.provenance_records
    )


def test_failed_task_identifies_base_and_effective_seed() -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    task = PerturbationTask(
        base=base,
        base_structure_id="wrong-id",
        settings=settings,
        counts=PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0),
        seed=909,
    )

    with pytest.raises(PerturbationTaskError, match="base=wrong-id, seed=909"):
        PerturbationCoordinator(settings=settings)._execute([task], 1)
