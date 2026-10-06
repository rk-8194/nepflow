"""Coordinator execution, defect placement, and worker failure contracts."""

import pickle
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

import nepflow.stages.generation.perturbations.coordinator as coordinator_module
from nepflow.domain.identities import calculate_structure_id
from nepflow.stages.generation.perturbations.coordinator import (
    PerturbationCoordinator,
    PerturbationTaskError,
    execute_perturbation_task,
    family_applies_to_base,
)
from nepflow.stages.generation.perturbations.defects import (
    gas_in_vacancy,
    gas_interstitials,
    interstitials,
    vacancies,
    vacancy_interstitial,
)
from nepflow.stages.generation.perturbations.models import (
    PERTURBATION_FAMILY_SOURCE_FIELDS,
    PerturbationCounts,
    PerturbationRejection,
    PerturbationSettings,
    PerturbationTask,
    derive_child_seed,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.validation import validate_generated_candidate


def base_atoms(*, source: str = "coordinator-fixture", seed_id: str = "seed_000001") -> Atoms:
    atoms = Atoms(
        "Si4",
        positions=[[2, 2, 2], [6, 2, 2], [2, 6, 2], [2, 2, 6]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    atoms.info.update(
        {
            "seed_id": seed_id,
            "source": source,
            "elements": ["Si"],
            "configurational_type": "test_base",
        }
    )
    return atoms


def scoped_base(source: str) -> Atoms:
    atoms = base_atoms(source=source, seed_id=f"seed-{source}")
    atoms.info["configurational_type"] = source
    return atoms


def test_child_seed_derivation_is_stable_and_namespaced() -> None:
    base_id = calculate_structure_id(base_atoms())

    assert derive_child_seed(base_id, 17, "vacancy", 0) == derive_child_seed(
        base_id, 17, "vacancy", 0
    )
    assert derive_child_seed(base_id, 17, "vacancy", 0) != derive_child_seed(
        base_id, 17, "vacancy", 1
    )
    assert derive_child_seed(base_id, 17, "vacancy", 0) != derive_child_seed(
        base_id, 17, "interstitial", 0
    )
    assert derive_child_seed(base_id, 17, "vacancy", 0) != derive_child_seed(
        base_id, 18, "vacancy", 0
    )
    assert 0 <= derive_child_seed(base_id, 17, "vacancy", 0) < 2**32


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda atoms: atoms.positions.__setitem__((0, 0), np.nan), "nonfinite_positions"),
        (
            lambda atoms: atoms.positions.__setitem__((1, 1), np.inf),
            "nonfinite_positions",
        ),
        (
            lambda atoms: atoms.set_cell([[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 12.0]]),
            "singular_periodic_cell",
        ),
        (
            lambda atoms: atoms.positions.__setitem__((1, slice(None)), [2.1, 2.0, 2.0]),
            "too_close_atoms",
        ),
    ],
)
def test_generated_candidate_validation_records_geometry_failures(mutate, reason: str) -> None:
    candidate = base_atoms()
    mutate(candidate)

    issue = validate_generated_candidate(
        candidate,
        base_atoms(),
        PerturbationSettings(target_n_atoms=4),
        "unperturbed",
    )

    assert issue is not None
    assert issue.reason == reason


def test_compressed_volume_state_uses_a_relaxed_geometry_policy() -> None:
    reference = base_atoms()
    candidate = reference.copy()
    candidate.set_cell(reference.cell * 0.2, scale_atoms=True)

    assert (
        validate_generated_candidate(
            candidate,
            reference,
            PerturbationSettings(target_n_atoms=4),
            "volume_profile",
        )
        is None
    )


def test_partial_interstitial_placement_is_rejected_with_counts() -> None:
    reference = base_atoms()
    candidate = reference.copy()
    candidate.append(Atoms("H", positions=[[8.0, 8.0, 8.0]], cell=reference.cell, pbc=True)[0])
    candidate.info.update(
        {
            "requested_n_interstitials": 2,
            "realised_n_interstitials": 1,
            "n_interstitials": 1,
        }
    )

    issue = validate_generated_candidate(
        candidate,
        reference,
        PerturbationSettings(target_n_atoms=4, interstitial_d_min=1.0),
        "interstitial",
    )

    assert issue is not None
    assert issue.reason == "partial_interstitial_placement"
    assert issue.evidence == {"requested": 2, "realised": 1}


def test_rejection_record_serializes_and_exposes_base_alias() -> None:
    rejection = PerturbationRejection(
        parent_structure_id="base-id",
        family="interstitial",
        operation_id="base-id:interstitial:0",
        slot=0,
        reason="partial_interstitial_placement",
        evidence={"requested": 2, "realised": 1},
    )

    assert rejection.base_structure_id == "base-id"
    assert rejection.to_dict() == {
        "parent_structure_id": "base-id",
        "family": "interstitial",
        "operation_id": "base-id:interstitial:0",
        "slot": 0,
        "reason": "partial_interstitial_placement",
        "evidence": {"requested": 2, "realised": 1},
    }
    assert pickle.loads(pickle.dumps(rejection)) == rejection


def test_focused_defect_families_remain_independently_callable() -> None:
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
    assert [result[0].info["random_seed"] for result in outputs] == [
        derive_child_seed(calculate_structure_id(base), 17, family, 0)
        for family in (
            "vacancy",
            "interstitial",
            "gas_interstitial",
            "vacancy_interstitial",
            "gas_in_vacancy",
        )
    ]
    assert outputs[0][0].info["n_vacancies"] == 1
    assert outputs[1][0].info["n_interstitials"] >= 0
    assert outputs[2][0].info["n_gas_interstitials"] >= 0
    assert outputs[3][0].info["n_vacancies"] == 1
    assert outputs[4][0].info["n_gas_atoms"] >= 0


@pytest.mark.parametrize(
    ("family", "scope_field"),
    [
        ("volume_profile", "volume_sources"),
        ("elastic_stress", "elastic_sources"),
        ("rattled", "rattle_sources"),
        ("liquid", "liquid_sources"),
        ("vacancy", "vacancy_sources"),
        ("interstitial", "interstitial_sources"),
        ("gas_interstitial", "gas_interstitial_sources"),
        ("vacancy_interstitial", "vacancy_interstitial_sources"),
        ("gas_in_vacancy", "gas_in_vacancy_sources"),
    ],
)
def test_each_family_runs_only_on_its_configured_source(family: str, scope_field: str) -> None:
    settings = PerturbationSettings(**{scope_field: ("mp_phase",)})

    assert family_applies_to_base(family, scoped_base("mp_phase"), settings)
    assert not family_applies_to_base(family, scoped_base("sqs"), settings)


def test_all_source_scope_requires_a_configurational_type() -> None:
    base = base_atoms()
    del base.info["configurational_type"]

    assert not family_applies_to_base(
        "vacancy",
        base,
        PerturbationSettings(vacancy_sources=("all",)),
    )


def test_one_perturbation_task_represents_one_base_structure() -> None:
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(target_n_atoms=4, random_seed=21)
    )
    counts = PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0)

    tasks = coordinator._tasks([first, second], counts)

    assert tasks[0].base is first
    assert tasks[1].base is second
    assert [task.base_structure_id for task in tasks] == [
        calculate_structure_id(first),
        calculate_structure_id(second),
    ]
    assert all(not isinstance(task.base, (list, tuple)) for task in tasks)


def test_serial_and_parallel_candidates_are_ordered_and_scientifically_equal() -> None:
    bases = [
        base_atoms(source="first", seed_id="seed-first"),
        base_atoms(source="second", seed_id="seed-second"),
    ]
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
        bases, counts=counts, n_workers=1
    )
    parallel = PerturbationCoordinator(settings=settings).generate_candidates(
        bases, counts=counts, n_workers=2
    )

    assert len(serial) == len(parallel)
    for left, right in zip(serial, parallel):
        np.testing.assert_array_equal(left.numbers, right.numbers)
        np.testing.assert_allclose(left.positions, right.positions)
        np.testing.assert_allclose(left.cell.array, right.cell.array)
        assert left.info == right.info
    assert [item.info["seed_id"] for item in serial] == [
        "seed-first",
        "seed-first",
        "seed-second",
        "seed-second",
    ]

    task = PerturbationCoordinator(settings=settings)._tasks([bases[0]], counts)[0]
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


def test_task_result_keeps_accepted_and_rejected_attempts_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = base_atoms()
    accepted = base.copy()
    accepted.info["perturbation_type"] = "rattled"
    rejected = base.copy()
    rejected.info["perturbation_type"] = "rattled"
    rejected.positions[0, 0] = np.nan

    def fake_rattled(*args, **kwargs):
        del args, kwargs
        return [accepted, rejected]

    monkeypatch.setattr(coordinator_module, "rattled", fake_rattled)
    task = PerturbationTask(
        base=base,
        base_structure_id=calculate_structure_id(base),
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        ),
        counts=PerturbationCounts(n_rattled=2, n_vacancies=0, n_interstitials=0),
        seed=21,
    )

    result = execute_perturbation_task(task)

    assert len(result.candidates) == 1
    assert len(result.provenance_records) == 1
    assert len(result.rejected_attempts) == 1
    assert result.rejected_attempts[0].reason == "nonfinite_positions"
    restored = pickle.loads(pickle.dumps(result))
    assert [item.to_dict() for item in restored.rejected_attempts] == [
        item.to_dict() for item in result.rejected_attempts
    ]


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


def test_existing_candidate_family_order_is_locked_before_source_scoping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_generators = (
        ("volume_profile", "volume_profile"),
        ("elastic_stress_set", "elastic_stress"),
        ("rattled", "rattled"),
        ("liquid_snapshots", "liquid"),
        ("vacancies", "vacancy"),
        ("interstitials", "interstitial"),
        ("gas_interstitials", "gas_interstitial"),
        ("vacancy_interstitial", "vacancy_interstitial"),
        ("gas_in_vacancy", "gas_in_vacancy"),
    )

    def tagged_family(name: str):
        def generate(*args, **kwargs):
            del args, kwargs
            candidate = base_atoms()
            candidate.info["perturbation_type"] = name
            return [candidate]

        return generate

    for generator_name, family_name in family_generators:
        monkeypatch.setattr(
            coordinator_module,
            generator_name,
            tagged_family(family_name),
        )

    result = execute_perturbation_task(
        PerturbationTask(
            base=base_atoms(),
            base_structure_id=calculate_structure_id(base_atoms()),
            settings=PerturbationSettings(
                target_n_atoms=4,
                gas_elements=("H",),
                elastic_stress_enabled=True,
                liquid_enabled=True,
            ),
            counts=PerturbationCounts(
                n_rattled=1,
                n_liquid_configurations=1,
                n_liquid_snapshots=1,
                n_vacancies=1,
                n_interstitials=1,
                n_gas_interstitials=1,
                n_vacancy_interstitial=1,
                n_gas_in_vacancy=1,
            ),
            seed=21,
        )
    )

    assert [item.info["perturbation_type"] for item in result.candidates] == [
        "unperturbed",
        *(family_name for _generator_name, family_name in family_generators),
    ]


def _source_scoped_candidates(
    bases: list[Atoms],
    *,
    counts: PerturbationCounts,
    family_scopes: dict[str, tuple[str, ...]],
) -> list[Atoms]:
    """Build typed settings for one source-scoped coordinator contract."""

    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            random_seed=21,
            n_volume_points=0,
            elastic_stress_enabled=False,
            **{
                PERTURBATION_FAMILY_SOURCE_FIELDS[family]: scope
                for family, scope in family_scopes.items()
            },
        )
    )
    return coordinator.generate_candidates(
        bases,
        counts=counts,
        n_workers=1,
    )


def test_mp_phase_scoped_family_does_not_run_on_sqs() -> None:
    candidates = _source_scoped_candidates(
        [scoped_base("mp_phase"), scoped_base("sqs")],
        counts=PerturbationCounts(n_rattled=0, n_vacancies=1, n_interstitials=0),
        family_scopes={"vacancy": ("mp_phase",)},
    )

    assert [
        item.info["source"] for item in candidates if item.info["perturbation_type"] == "vacancy"
    ] == ["mp_phase"], "a family scoped to mp_phase must not run on an sqs base"


def test_sqs_scoped_family_does_not_run_on_random_solid_solution() -> None:
    candidates = _source_scoped_candidates(
        [scoped_base("sqs"), scoped_base("random_solid_solution")],
        counts=PerturbationCounts(n_rattled=0, n_vacancies=1, n_interstitials=0),
        family_scopes={"vacancy": ("sqs",)},
    )

    assert [
        item.info["source"] for item in candidates if item.info["perturbation_type"] == "vacancy"
    ] == ["sqs"], "a family scoped to sqs must not run on a random_solid_solution base"


def test_explicit_all_sources_scope_applies_to_every_eligible_base() -> None:
    candidates = _source_scoped_candidates(
        [scoped_base("mp_phase"), scoped_base("sqs"), scoped_base("random_solid_solution")],
        counts=PerturbationCounts(n_rattled=0, n_vacancies=1, n_interstitials=0),
        family_scopes={"vacancy": ("all",)},
    )

    assert [
        item.info["source"] for item in candidates if item.info["perturbation_type"] == "vacancy"
    ] == ["mp_phase", "sqs", "random_solid_solution"], (
        "an explicit all-sources scope must apply to every eligible base"
    )


def test_enabling_two_families_does_not_imply_chaining() -> None:
    base = scoped_base("mp_phase")
    candidates = _source_scoped_candidates(
        [base],
        counts=PerturbationCounts(n_rattled=1, n_vacancies=1, n_interstitials=0),
        family_scopes={"rattled": ("mp_phase",), "vacancy": ("mp_phase",)},
    )

    families = [item.info["perturbation_type"] for item in candidates]
    assert families.count("rattled") == 1
    assert families.count("vacancy") == 1
    assert all(
        item.info["generation_provenance"]["parent_structure_id"] == calculate_structure_id(base)
        for item in candidates
        if item.info["perturbation_type"] in {"rattled", "vacancy"}
    ), "enabling two families must not feed one family's candidates into the other"


def test_filtering_one_family_does_not_reorder_unaffected_candidates() -> None:
    bases = [scoped_base("mp_phase"), scoped_base("sqs")]
    counts = PerturbationCounts(n_rattled=1, n_vacancies=1, n_interstitials=0)
    unfiltered = _source_scoped_candidates(
        bases,
        counts=counts,
        family_scopes={"rattled": ("all",), "vacancy": ("all",)},
    )
    filtered = _source_scoped_candidates(
        bases,
        counts=counts,
        family_scopes={"rattled": ("all",), "vacancy": ("mp_phase",)},
    )

    def unaffected_order(candidates: list[Atoms]) -> list[tuple[str, str]]:
        return [
            (item.info["source"], item.info["perturbation_type"])
            for item in candidates
            if item.info["perturbation_type"] != "vacancy"
        ]

    assert unaffected_order(filtered) == unaffected_order(unfiltered), (
        "filtering one family must not reorder unaffected candidates"
    )
