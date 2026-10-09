"""Coordinator execution, defect placement, and worker failure contracts."""

import pickle
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import read

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
from nepflow.stages.generation.perturbations.magnetism.models import (
    MagneticGenerationResult,
    MagneticGenerationSummary,
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


class _HarnessFuture:
    """Synchronous future that records when the coordinator consumes a result."""

    def __init__(self, harness: "PerturbationExecutionHarness", function, task) -> None:
        self._harness = harness
        self._function = function
        self._task = task

    def result(self):
        if self._harness.first_result_submission_count is None:
            self._harness.first_result_submission_count = len(self._harness.submitted_task_ids)
        try:
            return self._function(self._task)
        finally:
            self._harness.resolved_task_ids.append(self._task.base_structure_id)
            self._harness.in_flight_count -= 1

    def cancel(self) -> bool:
        self._harness.cancelled_task_ids.append(self._task.base_structure_id)
        return True


class PerturbationExecutionHarness:
    """Record pool submission, resolution, and ordered result consumption.

    Futures execute only when consumed, so tests can prove that an earlier
    result is flushed before a later task is allowed to complete.
    """

    def __init__(self) -> None:
        self.worker_counts: list[int] = []
        self.submitted_task_ids: list[str] = []
        self.resolved_task_ids: list[str] = []
        self.in_flight_count = 0
        self.peak_in_flight_count = 0
        self.first_result_submission_count: int | None = None
        self.cancelled_task_ids: list[str] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def executor_factory(self, *, max_workers: int):
        self.worker_counts.append(max_workers)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        return False

    def submit(self, function, task):
        self.submitted_task_ids.append(task.base_structure_id)
        self.in_flight_count += 1
        self.peak_in_flight_count = max(self.peak_in_flight_count, self.in_flight_count)
        return _HarnessFuture(self, function, task)

    def shutdown(self, *, wait: bool, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _GlobalDefectLimitMagneticHarness:
    """Minimal magnetic expander proving the coordinator supplies one global stream."""

    def __init__(self, *, max_defect_parents: int) -> None:
        self.max_defect_parents = max_defect_parents
        self.calls: list[list[tuple[str, str]]] = []

    def expand_structures(self, structures):
        candidates = list(structures)
        self.calls.append(
            [
                (str(item.info["seed_id"]), str(item.info["perturbation_type"]))
                for item in candidates
            ]
        )
        seen_defect_parents = 0
        expanded = []
        for candidate in candidates:
            materialized = candidate.copy()
            if materialized.info["perturbation_type"] == "vacancy":
                if seen_defect_parents < self.max_defect_parents:
                    materialized.info["magnetic_ordering"] = "afm"
                seen_defect_parents += 1
            expanded.append(materialized)
        return MagneticGenerationResult(tuple(expanded), MagneticGenerationSummary())


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

    tasks = list(coordinator._tasks([first, second], counts))

    assert tasks[0].base is first
    assert tasks[1].base is second
    assert [task.base_structure_id for task in tasks] == [
        calculate_structure_id(first),
        calculate_structure_id(second),
    ]
    assert all(not isinstance(task.base, (list, tuple)) for task in tasks)


def test_serial_and_parallel_candidates_are_ordered_and_scientifically_equal() -> None:
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    bases = [first, second]
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

    task = next(PerturbationCoordinator(settings=settings)._tasks([bases[0]], counts))
    result = execute_perturbation_task(task)
    assert len(result.provenance_records) == len(result.candidates)


def test_serial_and_parallel_publication_lock_scopes_seeds_and_extxyz_bytes(tmp_path) -> None:
    """The bounded-streaming work must preserve this complete scientific baseline."""

    def bases() -> list[Atoms]:
        first = scoped_base("mp_phase")
        first.info["seed_id"] = "seed-mp"
        second = scoped_base("sqs")
        second.info["seed_id"] = "seed-sqs"
        second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
        return [first, second]

    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
        rattle_sources=("sqs",),
        vacancy_sources=("mp_phase",),
    )
    counts = PerturbationCounts(n_rattled=1, n_vacancies=1, n_interstitials=0)
    expected_tasks = list(PerturbationCoordinator(settings=settings)._tasks(bases(), counts))
    serial = PerturbationCoordinator(settings=settings)
    serial_path = serial.process(
        bases(),
        tmp_path / "serial",
        n_rattled=counts.n_rattled,
        n_vacancies=counts.n_vacancies,
        n_interstitials=counts.n_interstitials,
        n_workers=1,
    )
    parallel = PerturbationCoordinator(settings=settings)
    parallel_path = parallel.process(
        bases(),
        tmp_path / "parallel",
        n_rattled=counts.n_rattled,
        n_vacancies=counts.n_vacancies,
        n_interstitials=counts.n_interstitials,
        n_workers=2,
    )

    assert serial_path.read_bytes() == parallel_path.read_bytes()
    serial_candidates = read(str(serial_path), index=":")
    assert [
        (item.info["seed_id"], item.info["perturbation_type"]) for item in serial_candidates
    ] == [
        ("seed-mp", "unperturbed"),
        ("seed-mp", "vacancy"),
        ("seed-sqs", "unperturbed"),
        ("seed-sqs", "rattled"),
    ]
    assert [record.provenance.random_seed for record in serial.get_provenance_records()] == [
        None,
        derive_child_seed(expected_tasks[0].base_structure_id, expected_tasks[0].seed, "vacancy"),
        None,
        derive_child_seed(expected_tasks[1].base_structure_id, expected_tasks[1].seed, "rattled"),
    ]
    assert [record.to_dict() for record in serial.get_provenance_records()] == [
        record.to_dict() for record in parallel.get_provenance_records()
    ]
    assert (
        serial.get_summary()
        == parallel.get_summary()
        == {
            "total": 4,
            "by_type": {"unperturbed": 2, "vacancy": 1, "rattled": 1},
            "by_config": {"mp_phase": 2, "sqs": 2},
            "duplicate_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {},
            "rejections_by_family": {},
        }
    )


def test_execution_harness_bounds_parallel_window_and_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow first task must exert backpressure without reordering results."""

    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    coordinator = PerturbationCoordinator(settings=settings)
    bases = [base_atoms(source=f"base-{index}") for index in range(5)]
    for index, base in enumerate(bases):
        base.set_cell(np.diag([12.0 + index, 12.0, 12.0]), scale_atoms=False)
    tasks = list(
        coordinator._tasks(
            bases,
            PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0),
        )
    )
    harness = PerturbationExecutionHarness()
    monkeypatch.setattr(coordinator_module, "ProcessPoolExecutor", harness.executor_factory)

    results = list(coordinator._execute(tasks, n_workers=2))

    expected_ids = [task.base_structure_id for task in tasks]
    assert harness.worker_counts == [2]
    assert harness.submitted_task_ids == expected_ids
    assert harness.resolved_task_ids == expected_ids
    assert [result.task.base_structure_id for result in results] == expected_ids
    assert harness.peak_in_flight_count <= 4
    assert harness.first_result_submission_count == 4
    assert harness.first_result_submission_count < len(tasks)
    assert harness.shutdown_calls == [(True, False)]


@pytest.mark.parametrize("n_workers", (1, 2))
def test_process_flushes_first_result_before_later_task_completes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    n_workers: int,
) -> None:
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
    )
    first_result_flushed = False
    original_execute = coordinator_module.execute_perturbation_task
    original_flush = coordinator._flush

    def controlled_worker(task):
        if task.base.info["source"] == "second":
            assert first_result_flushed, (
                "the first result must flush before the later task finishes"
            )
        return original_execute(task)

    def record_flush(candidates, *, output_handle=None):
        nonlocal first_result_flushed
        result = original_flush(candidates, output_handle=output_handle)
        if any(candidate.info["source"] == "first" for candidate in candidates):
            first_result_flushed = True
        return result

    monkeypatch.setattr(coordinator_module, "execute_perturbation_task", controlled_worker)
    monkeypatch.setattr(coordinator, "_flush", record_flush)
    if n_workers == 2:
        harness = PerturbationExecutionHarness()
        monkeypatch.setattr(coordinator_module, "ProcessPoolExecutor", harness.executor_factory)

    coordinator.process(
        [first, second],
        tmp_path,
        n_rattled=0,
        n_vacancies=0,
        n_interstitials=0,
        n_workers=n_workers,
    )

    assert first_result_flushed


def test_auto_workers_uses_the_same_bounded_parallel_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    bases = [base_atoms(source=f"base-{index}") for index in range(5)]
    for index, base in enumerate(bases):
        base.set_cell(np.diag([12.0 + index, 12.0, 12.0]), scale_atoms=False)
    harness = PerturbationExecutionHarness()
    monkeypatch.setattr(coordinator_module.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(coordinator_module, "ProcessPoolExecutor", harness.executor_factory)

    candidates = PerturbationCoordinator(settings=settings).generate_candidates(
        bases,
        counts=PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0),
        n_workers=0,
    )

    assert len(candidates) == len(bases)
    assert harness.worker_counts == [2]
    assert harness.peak_in_flight_count <= 4


def test_parallel_failure_cancels_pending_futures_with_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    bases = [base_atoms(source=f"base-{index}") for index in range(4)]
    for index, base in enumerate(bases):
        base.set_cell(np.diag([12.0 + index, 12.0, 12.0]), scale_atoms=False)
    coordinator = PerturbationCoordinator(settings=settings)
    tasks = list(
        coordinator._tasks(
            bases,
            PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0),
        )
    )
    harness = PerturbationExecutionHarness()
    original_execute = coordinator_module.execute_perturbation_task

    def fail_second_task(task):
        if task.base.info["source"] == "base-1":
            raise RuntimeError("controlled worker failure")
        return original_execute(task)

    monkeypatch.setattr(coordinator_module, "ProcessPoolExecutor", harness.executor_factory)
    monkeypatch.setattr(coordinator_module, "execute_perturbation_task", fail_second_task)

    with pytest.raises(
        PerturbationTaskError,
        match=f"base={tasks[1].base_structure_id}, seed={tasks[1].seed}",
    ):
        list(coordinator._execute(tasks, n_workers=2))

    assert harness.cancelled_task_ids == [task.base_structure_id for task in tasks[2:]]
    assert harness.shutdown_calls == [(False, True)]


def test_magnetic_disabled_baseline_does_not_add_magnetic_state(tmp_path) -> None:
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
    )

    output_path = coordinator.process(
        [base_atoms()],
        tmp_path,
        n_rattled=0,
        n_vacancies=1,
        n_interstitials=0,
        n_workers=1,
    )

    assert "magnetic" not in coordinator.get_summary()
    assert all("magnetic_ordering" not in item.info for item in read(str(output_path), index=":"))


def test_magnetic_harness_receives_global_ordered_defect_parents(tmp_path) -> None:
    """A real magnetic defect-parent cap must see every structural parent once."""

    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    magnetic = _GlobalDefectLimitMagneticHarness(max_defect_parents=1)
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        ),
        magnetic_generator=magnetic,
    )

    output_path = coordinator.process(
        [first, second],
        tmp_path,
        n_rattled=0,
        n_vacancies=1,
        n_interstitials=0,
        n_workers=1,
    )

    assert magnetic.calls == [
        [
            ("seed-first", "unperturbed"),
            ("seed-first", "vacancy"),
            ("seed-second", "unperturbed"),
            ("seed-second", "vacancy"),
        ]
    ]
    vacancies = [
        item
        for item in read(str(output_path), index=":")
        if item.info["perturbation_type"] == "vacancy"
    ]
    assert [item.info.get("magnetic_ordering") for item in vacancies] == ["afm", None]


def test_final_deduplication_keeps_unperturbed_representative_and_all_paths() -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=1,
        volume_scale_range=(1.0, 1.0),
        elastic_stress_enabled=False,
    )
    counts = PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0)
    coordinator = PerturbationCoordinator(settings=settings)

    candidates = coordinator.generate_candidates([base], counts=counts, n_workers=1)

    assert len(candidates) == 1
    assert candidates[0].info["perturbation_type"] == "unperturbed"
    assert coordinator.get_summary()["duplicate_count"] == 1
    records = coordinator.get_provenance_records()
    assert [record.provenance.operation_id for record in records] == [
        f"{calculate_structure_id(base)}:unperturbed",
        f"{calculate_structure_id(base)}:volume:0",
    ]
    assert [record.structure_id for record in records] == [
        calculate_structure_id(base),
        calculate_structure_id(base),
    ]


def test_final_deduplication_counts_published_candidates_and_is_parallel_stable(tmp_path) -> None:
    base = base_atoms()
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=1,
        volume_scale_range=(1.0, 1.0),
        elastic_stress_enabled=False,
    )
    kwargs = {
        "n_rattled": 0,
        "n_vacancies": 0,
        "n_interstitials": 0,
        "n_workers": 1,
    }
    serial = PerturbationCoordinator(settings=settings)
    serial_path = serial.process([base], tmp_path / "serial", **kwargs)

    parallel = PerturbationCoordinator(settings=settings)
    parallel_kwargs = dict(kwargs, n_workers=2)
    parallel_path = parallel.process([base], tmp_path / "parallel", **parallel_kwargs)

    assert serial.get_summary()["total"] == 1
    assert serial.get_summary()["duplicate_count"] == 1
    assert serial_path.read_bytes() == parallel_path.read_bytes()
    assert [record.provenance.operation_id for record in serial.get_provenance_records()] == [
        record.provenance.operation_id for record in parallel.get_provenance_records()
    ]
    assert len(read(str(serial_path), index=":")) == 1


def test_publication_keeps_task_order_without_reading_prior_batches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("publication must not reread the accumulated artifact")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    output_path = PerturbationCoordinator(settings=settings).process(
        [first, second],
        tmp_path,
        n_rattled=0,
        n_vacancies=0,
        n_interstitials=0,
        n_workers=1,
    )
    monkeypatch.undo()

    published = read(str(output_path), index=":")
    assert [item.info["seed_id"] for item in published] == ["seed-first", "seed-second"]
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_failed_late_publication_preserves_previous_complete_artifact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "generated_structures.xyz"
    output_path.write_bytes(b"complete-candidate-artifact")
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
    )
    original_flush = coordinator._flush
    calls = 0

    def fail_on_second_batch(candidates, *, output_handle=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled publication failure")
        return original_flush(candidates, output_handle=output_handle)

    monkeypatch.setattr(coordinator, "_flush", fail_on_second_batch)
    with pytest.raises(RuntimeError, match="controlled publication failure"):
        coordinator.process(
            [first, second],
            tmp_path,
            n_rattled=0,
            n_vacancies=0,
            n_interstitials=0,
            n_workers=1,
        )

    assert output_path.read_bytes() == b"complete-candidate-artifact"
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_failed_later_task_preserves_complete_artifact_and_reports_its_seed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic append-to-temporary-file publication is the completed #85 prerequisite."""

    output_path = tmp_path / "generated_structures.xyz"
    output_path.write_bytes(b"complete-candidate-artifact")
    first = base_atoms(source="first", seed_id="seed-first")
    second = base_atoms(source="second", seed_id="seed-second")
    second.set_cell(np.diag([12.5, 12.0, 12.0]), scale_atoms=False)
    settings = PerturbationSettings(
        target_n_atoms=4,
        random_seed=21,
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    coordinator = PerturbationCoordinator(settings=settings)
    expected_later_task = list(
        PerturbationCoordinator(settings=settings)._tasks(
            [first, second],
            PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0),
        )
    )[1]
    original_execute = coordinator_module.execute_perturbation_task

    def fail_second_task(task):
        if task.base.info["source"] == "second":
            raise RuntimeError("controlled later task failure")
        return original_execute(task)

    monkeypatch.setattr(coordinator_module, "execute_perturbation_task", fail_second_task)

    with pytest.raises(
        PerturbationTaskError,
        match=(
            f"base={expected_later_task.base_structure_id}, seed={expected_later_task.seed}: "
            "controlled later task failure"
        ),
    ):
        coordinator.process(
            [first, second],
            tmp_path,
            n_rattled=0,
            n_vacancies=0,
            n_interstitials=0,
            n_workers=1,
        )

    assert output_path.read_bytes() == b"complete-candidate-artifact"
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_zero_candidate_publication_is_an_explicit_empty_artifact(tmp_path) -> None:
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
    )

    output_path = coordinator.process(
        [],
        tmp_path,
        n_rattled=0,
        n_vacancies=0,
        n_interstitials=0,
        n_workers=1,
    )

    assert output_path.is_file()
    assert output_path.read_bytes() == b""
    assert coordinator.get_summary()["total"] == 0
    assert coordinator.get_provenance_records() == ()
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


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
    task = next(PerturbationCoordinator(settings=settings)._tasks([base], counts))

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

    assert [candidate.info["perturbation_type"] for candidate in result.candidates] == [
        "unperturbed",
        "rattled",
    ]
    assert len(result.provenance_records) == 2
    assert len(result.rejected_attempts) == 1
    assert result.rejected_attempts[0].family == "rattled"
    assert result.rejected_attempts[0].reason == "nonfinite_positions"
    restored = pickle.loads(pickle.dumps(result))
    assert [item.to_dict() for item in restored.rejected_attempts] == [
        item.to_dict() for item in result.rejected_attempts
    ]


def test_rejected_attempts_are_reported_without_publishing_invalid_candidates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = base_atoms()
    rejected = base.copy()
    rejected.info["perturbation_type"] = "rattled"
    rejected.positions[0, 0] = np.nan

    def fake_rattled(*args, **kwargs):
        del args, kwargs
        return [rejected]

    monkeypatch.setattr(coordinator_module, "rattled", fake_rattled)
    coordinator = PerturbationCoordinator(
        settings=PerturbationSettings(
            target_n_atoms=4,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
    )

    output_path = coordinator.process(
        [base],
        tmp_path,
        n_rattled=1,
        n_vacancies=0,
        n_interstitials=0,
        n_workers=1,
    )

    assert [item.info["perturbation_type"] for item in read(str(output_path), index=":")] == [
        "unperturbed"
    ]
    assert coordinator.get_summary()["rejected_count"] == 1
    assert coordinator.get_summary()["rejected_reason_counts"] == {"nonfinite_positions": 1}
    assert coordinator.get_summary()["rejections_by_family"] == {
        "rattled": {"nonfinite_positions": 1}
    }


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
        list(PerturbationCoordinator(settings=settings)._execute([task], 1))


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
            if name == "liquid":
                candidate.info.update(
                    {
                        "liquid_method": "ase_langevin_lj",
                        "liquid_fidelity": "geometry_disorder_only_not_material_specific",
                        "liquid_temperature_k": 3000.0,
                        "liquid_timestep_fs": 1.0,
                        "liquid_friction": 0.02,
                        "liquid_equilibration_steps": 200,
                        "liquid_steps_between_snapshots": 100,
                        "liquid_configuration_index": 0,
                        "liquid_snapshot_index": 0,
                        "liquid_snapshot_step": 300,
                        "liquid_effective_child_seed": 1,
                        "parent_structure_id": calculate_structure_id(candidate),
                        "source_composition": {"Si": 1.0},
                        "liquid_source_parent_structure_id": calculate_structure_id(candidate),
                        "liquid_source_composition": {"Si": 1.0},
                    }
                )
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
