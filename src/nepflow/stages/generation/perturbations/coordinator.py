"""Small typed coordinator for deterministic perturbation workers."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from io import StringIO
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from ase.io import write

from nepflow.config.models import ALL_SOURCES
from nepflow.domain.identities import (
    annotate_candidate_id,
    annotate_structure_ids,
    calculate_structure_id,
)
from nepflow.domain.structures import GeneratedStructureRecord
from nepflow.stages.generation.supercell import build_target_supercell
from nepflow.stages.generation.validation import (
    CandidateValidationIssue,
    validate_generated_candidate,
)

from .defects import (
    antisites,
    gas_in_vacancy,
    gas_interstitials,
    interstitials,
    substitutions,
    vacancies,
    vacancy_interstitial,
)
from .displacements import rattled
from .elastic import elastic_stress_set
from .grain_boundaries import grain_boundaries
from .liquid import liquid_snapshots
from .magnetism import MagneticGenerationSummary, MagneticGenerator
from .models import (
    PerturbationCounts,
    PerturbationRejection,
    PerturbationSettings,
    PerturbationTask,
    PerturbationTaskResult,
)
from .provenance import annotate_generation_provenance
from .surfaces import surfaces
from .volume import volume_profile

logger = logging.getLogger(__name__)
_AUTO_WORKER_CAP = 8
_HEARTBEAT_INTERVAL_SECONDS = 30.0
_PERTURBATION_FAMILIES = frozenset(
    {
        "unperturbed",
        "volume_profile",
        "elastic_stress",
        "rattled",
        "liquid",
        "vacancy",
        "interstitial",
        "gas_interstitial",
        "substitution",
        "antisite",
        "vacancy_interstitial",
        "gas_in_vacancy",
        "surface",
        "grain_boundary",
    }
)


class PerturbationTaskError(RuntimeError):
    """A worker failure attributed to one exact base and effective seed."""

    def __init__(self, task: PerturbationTask, cause: BaseException) -> None:
        self.base_structure_id = task.base_structure_id
        self.seed = task.seed
        self.cause = cause
        super().__init__(
            f"Perturbation task failed for base={task.base_structure_id}, seed={task.seed}: {cause}"
        )


def family_applies_to_base(
    family: str,
    base: Any,
    settings: PerturbationSettings,
) -> bool:
    """Return whether one derived family may use the supplied base source."""

    scope = settings.sources_for_family(family)
    source = getattr(base, "info", {}).get("configurational_type")
    if not isinstance(source, str) or not source.strip():
        return False
    normalized_source = source.strip().lower()
    return ALL_SOURCES in scope or normalized_source in scope


def execute_perturbation_task(task: PerturbationTask) -> PerturbationTaskResult:
    """Execute one complete task; this function is process-pool picklable."""

    actual_base_id = calculate_structure_id(task.base)
    if actual_base_id != task.base_structure_id:
        raise ValueError(
            "Perturbation task base identity mismatch: "
            f"expected {task.base_structure_id}, got {actual_base_id}"
        )
    settings = task.settings
    supercell = build_target_supercell(
        task.base,
        target_n_atoms=settings.target_n_atoms,
    )
    if supercell is None:
        raise RuntimeError("target supercell construction returned no structure")

    candidate_records: dict[int, Any] = {}

    def annotate(
        candidate: Any,
        base: Any,
        family: str,
        *,
        random_seed: int | None = None,
        parameters: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> Any:
        record = annotate_generation_provenance(
            candidate,
            base,
            family,
            random_seed=random_seed,
            parameters=parameters,
            operation_id=(
                f"{task.base_structure_id}:{operation_id}" if operation_id is not None else None
            ),
        )
        candidate_records[id(candidate)] = record
        return record

    output: list[Any] = []
    equilibrium = supercell.copy()
    annotate(
        equilibrium,
        task.base,
        "unperturbed",
        operation_id="unperturbed",
    )
    output.append(equilibrium)
    if settings.n_volume_points > 0 and family_applies_to_base(
        "volume_profile", task.base, settings
    ):
        output.extend(volume_profile(supercell, task.base, settings, annotate))
    if (
        settings.elastic_stress_enabled
        and settings.elastic_strain_amplitudes
        and family_applies_to_base("elastic_stress", task.base, settings)
    ):
        output.extend(elastic_stress_set(supercell, task.base, settings, annotate))
    if task.counts.n_rattled > 0 and family_applies_to_base("rattled", task.base, settings):
        output.extend(
            rattled(supercell, task.base, task.counts.n_rattled, settings, task.seed, annotate)
        )
    if (
        settings.liquid_enabled
        and task.counts.n_liquid_configurations > 0
        and task.counts.n_liquid_snapshots > 0
        and family_applies_to_base("liquid", task.base, settings)
    ):
        output.extend(
            liquid_snapshots(
                supercell,
                task.base,
                task.counts.n_liquid_configurations,
                task.counts.n_liquid_snapshots,
                settings,
                task.seed,
                annotate,
            )
        )
    if task.counts.n_vacancies > 0 and family_applies_to_base("vacancy", task.base, settings):
        output.extend(
            vacancies(
                supercell,
                task.base,
                task.counts.n_vacancies,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if task.counts.n_interstitials > 0 and family_applies_to_base(
        "interstitial", task.base, settings
    ):
        output.extend(
            interstitials(
                supercell,
                task.base,
                task.counts.n_interstitials,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if (
        settings.gas_elements
        and task.counts.n_gas_interstitials > 0
        and family_applies_to_base("gas_interstitial", task.base, settings)
    ):
        output.extend(
            gas_interstitials(
                supercell,
                task.base,
                task.counts.n_gas_interstitials,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if task.counts.n_substitutions > 0 and family_applies_to_base(
        "substitution", task.base, settings
    ):
        output.extend(
            substitutions(
                supercell,
                task.base,
                task.counts.n_substitutions,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if task.counts.n_antisites > 0 and family_applies_to_base("antisite", task.base, settings):
        output.extend(
            antisites(
                supercell,
                task.base,
                task.counts.n_antisites,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if (
        settings.gas_elements
        and task.counts.n_vacancy_interstitial > 0
        and family_applies_to_base("vacancy_interstitial", task.base, settings)
    ):
        output.extend(
            vacancy_interstitial(
                supercell,
                task.base,
                task.counts.n_vacancy_interstitial,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if (
        settings.gas_elements
        and task.counts.n_gas_in_vacancy > 0
        and family_applies_to_base("gas_in_vacancy", task.base, settings)
    ):
        output.extend(
            gas_in_vacancy(
                supercell,
                task.base,
                task.counts.n_gas_in_vacancy,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if settings.surface_enabled and family_applies_to_base("surface", task.base, settings):
        # ``n_surfaces`` is a legacy compatibility input.  Preserve its old
        # single-orientation behaviour for direct callers, but never let it
        # truncate a multi-orientation request.  A zero legacy count means
        # "use the configured surface plan" in the new contract.
        surface_count: int | None = None
        if len(settings.surface_miller_indices) == 1 and task.counts.n_surfaces > 0:
            surface_count = task.counts.n_surfaces
        output.extend(
            surfaces(
                task.base,
                task.base,
                surface_count,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    if (
        settings.grain_boundary_enabled
        and task.counts.n_grain_boundaries > 0
        and family_applies_to_base("grain_boundary", task.base, settings)
    ):
        output.extend(
            grain_boundaries(
                task.base,
                task.base,
                task.counts.n_grain_boundaries,
                settings,
                None,
                annotate,
                seed=task.seed,
            )
        )
    accepted: list[Any] = []
    provenance_records: list[Any] = []
    rejected_attempts: list[PerturbationRejection] = []
    for index, candidate in enumerate(output):
        family = candidate.info.get("perturbation_type", "unknown")
        issue: CandidateValidationIssue | None
        if family not in _PERTURBATION_FAMILIES:
            issue = CandidateValidationIssue(
                "unknown_perturbation_family",
                {"family": family},
            )
        else:
            issue = validate_generated_candidate(candidate, supercell, settings, family)

        record = candidate_records.get(id(candidate))
        if issue is None and record is None:
            try:
                record = annotate_generation_provenance(
                    candidate,
                    task.base,
                    family,
                    random_seed=candidate.info.get("random_seed"),
                    operation_id=f"{task.base_structure_id}:{family}:{index}",
                )
            except Exception as exc:
                issue = CandidateValidationIssue(
                    "provenance_annotation_failed",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        if issue is None:
            accepted.append(candidate)
            if record is not None:
                provenance_records.append(record)
            continue

        operation_id = f"{task.base_structure_id}:{family}:{index}"
        if record is not None:
            operation_id = str(record.provenance.operation_id)
        else:
            provenance = candidate.info.get("generation_provenance", {})
            if isinstance(provenance, dict) and provenance.get("operation_id"):
                operation_id = str(provenance["operation_id"])
        slot: int | str | None = _rejection_slot(operation_id, index)
        rejected_attempts.append(
            PerturbationRejection(
                parent_structure_id=task.base_structure_id,
                family=str(family),
                operation_id=operation_id,
                slot=slot,
                reason=issue.reason,
                evidence=issue.evidence,
            )
        )

    return PerturbationTaskResult(
        task=task,
        candidates=tuple(accepted),
        provenance_records=tuple(provenance_records),
        rejected_attempts=tuple(rejected_attempts),
    )


def _rejection_slot(operation_id: str, fallback: int) -> int | str:
    """Extract the deterministic family slot from an operation identifier."""

    value = operation_id.rsplit(":", 1)[-1]
    try:
        return int(value)
    except ValueError:
        return value if value else fallback


class PerturbationCoordinator:
    """Enumerate tasks, execute them, and stream ordered candidate results."""

    def __init__(
        self,
        settings: PerturbationSettings | None = None,
        *,
        magnetic_generator: MagneticGenerator | None = None,
        **kwargs: Any,
    ) -> None:
        if settings is not None and kwargs:
            raise TypeError("pass either typed settings or legacy keyword settings")
        self.settings = settings or PerturbationSettings(**kwargs)
        self.magnetic_generator = magnetic_generator
        self.rng = np.random.RandomState(self.settings.random_seed)
        self._total = 0
        self._by_type: dict[str, int] = {}
        self._by_config: dict[str, int] = {}
        self._rejected_attempts: list[PerturbationRejection] = []
        self._provenance_records: tuple[GeneratedStructureRecord, ...] = ()
        self._duplicate_count = 0
        self._output_file: Path | None = None
        self._magnetic_summary: MagneticGenerationSummary | None = None
        self._task_elapsed_seconds: dict[tuple[str, int], float] = {}

    @staticmethod
    def _available_cpu_count() -> int:
        """Return the smallest reliable local CPU allocation visible to the process."""

        limits = [max(1, os.cpu_count() or 1)]
        try:
            limits.append(max(1, len(os.sched_getaffinity(0))))
        except (AttributeError, OSError):
            pass
        for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
            value = os.environ.get(variable, "").split("(", 1)[0].strip()
            if value.isdigit() and int(value) > 0:
                limits.append(int(value))
        for quota_path, period_path in (
            ("/sys/fs/cgroup/cpu.max", None),
            ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ):
            try:
                if period_path is None:
                    quota, period = Path(quota_path).read_text(encoding="utf-8").split()[:2]
                    if quota == "max":
                        continue
                else:
                    quota = Path(quota_path).read_text(encoding="utf-8").strip()
                    period = Path(period_path).read_text(encoding="utf-8").strip()
                quota_value = int(quota)
                period_value = int(period)
                if quota_value > 0 and period_value > 0:
                    limits.append(max(1, math.floor(quota_value / period_value)))
            except (OSError, ValueError, IndexError):
                continue
        return min(limits)

    @classmethod
    def _effective_worker_count(cls, requested_workers: int, task_count: int) -> int:
        """Apply the documented auto-worker allocation and runnable-task bounds."""

        if requested_workers < 0:
            raise ValueError("n_workers must be non-negative or at least one")
        runnable_tasks = max(1, task_count)
        if requested_workers == 0:
            return min(cls._available_cpu_count(), _AUTO_WORKER_CAP, runnable_tasks)
        return min(requested_workers, runnable_tasks)

    @staticmethod
    def _planned_families(
        counts: PerturbationCounts,
        settings: PerturbationSettings,
    ) -> tuple[str, ...]:
        """Return the enabled task-family plan for operational logging only."""

        families = ["unperturbed"]
        if settings.n_volume_points > 0:
            families.append("volume_profile")
        if settings.elastic_stress_enabled and settings.elastic_strain_amplitudes:
            families.append("elastic_stress")
        if counts.n_rattled > 0:
            families.append("rattled")
        if settings.liquid_enabled and counts.n_liquid_configurations > 0:
            families.append("liquid")
        for family, count in (
            ("vacancy", counts.n_vacancies),
            ("interstitial", counts.n_interstitials),
            ("gas_interstitial", counts.n_gas_interstitials),
            ("substitution", counts.n_substitutions),
            ("antisite", counts.n_antisites),
            ("vacancy_interstitial", counts.n_vacancy_interstitial),
            ("gas_in_vacancy", counts.n_gas_in_vacancy),
            ("grain_boundary", counts.n_grain_boundaries),
        ):
            if count > 0:
                families.append(family)
        if settings.surface_enabled:
            families.append("surface")
        return tuple(families)

    def _counts(self, **kwargs: int) -> PerturbationCounts:
        return PerturbationCounts(**kwargs)

    def _tasks(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> Iterator[PerturbationTask]:
        """Yield tasks in canonical base order with their stable RNG seeds."""

        task_settings = settings or self.settings
        for base in base_structures:
            yield PerturbationTask(
                base=base,
                base_structure_id=calculate_structure_id(base),
                settings=task_settings,
                counts=counts,
                seed=int(self.rng.randint(0, 2**31)),
            )

    def generate_candidates(
        self,
        base_structures: list[Any],
        *,
        counts: PerturbationCounts,
        n_workers: int = 1,
    ) -> list[Any]:
        """Generate candidates without persistence, preserving task order."""

        n_workers = self._effective_worker_count(n_workers, len(base_structures))
        self._rejected_attempts = []
        self._provenance_records = ()
        self._duplicate_count = 0
        self._magnetic_summary = None
        self._task_elapsed_seconds = {}

        def results_with_rejections() -> Iterator[PerturbationTaskResult]:
            for result in self._execute(self._tasks(base_structures, counts), n_workers):
                self._rejected_attempts.extend(result.rejected_attempts)
                yield result

        candidates, records = self._deduplicate_results(results_with_rejections())
        if self.magnetic_generator is not None:
            magnetic_result = self.magnetic_generator.expand_structures(candidates)
            candidates = list(magnetic_result.candidates)
            self._magnetic_summary = magnetic_result.summary
        self._provenance_records = records
        return candidates

    def process(
        self,
        base_structures: list[Any],
        output_dir: Path,
        n_rattled: int = 10,
        n_liquid_configurations: int = 0,
        n_liquid_snapshots: int = 0,
        n_vacancies: int = 10,
        n_interstitials: int = 10,
        n_gas_interstitials: int = 0,
        n_substitutions: int = 0,
        n_antisites: int = 0,
        n_vacancy_interstitial: int = 0,
        n_gas_in_vacancy: int = 0,
        n_surfaces: int = 0,
        n_grain_boundaries: int = 0,
        n_workers: int = 0,
    ) -> Path:
        """Run typed tasks and persist candidates in deterministic base order."""

        n_workers = self._effective_worker_count(n_workers, len(base_structures))
        self._total = 0
        self._by_type = {}
        self._by_config = {}
        self._rejected_attempts = []
        self._provenance_records = ()
        self._duplicate_count = 0
        self._magnetic_summary = None
        self._task_elapsed_seconds = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "generated_structures.xyz"
        self._output_file = output_file
        counts = self._counts(
            n_rattled=n_rattled,
            n_liquid_configurations=n_liquid_configurations,
            n_liquid_snapshots=n_liquid_snapshots,
            n_vacancies=n_vacancies,
            n_interstitials=n_interstitials,
            n_gas_interstitials=n_gas_interstitials,
            n_substitutions=n_substitutions,
            n_antisites=n_antisites,
            n_vacancy_interstitial=n_vacancy_interstitial,
            n_gas_in_vacancy=n_gas_in_vacancy,
            # Surface execution is orientation/termination driven. Retain the
            # parameter for API compatibility, but do not pass its legacy
            # global cap into the typed task plan.
            n_surfaces=0,
            n_grain_boundaries=n_grain_boundaries,
        )
        atom_counts = [len(base) for base in base_structures]
        atom_count_summary = "0" if not atom_counts else f"{min(atom_counts)}-{max(atom_counts)}"
        family_plan = ", ".join(self._planned_families(counts, self.settings))
        logger.info(
            "Starting perturbations: tasks=%s, workers=%s, submission_window=%s, "
            "base_atoms=%s, families=%s, output=%s",
            len(base_structures),
            n_workers,
            1 if n_workers == 1 else 2 * n_workers,
            atom_count_summary,
            family_plan,
            output_file,
        )
        temporary_path: Path | None = None
        file_descriptor: int | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_file.name}.",
                suffix=".tmp",
                dir=output_dir,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "wb") as output_handle:
                file_descriptor = None
                seen_structure_ids: set[str] = set()
                seen_operation_ids: set[tuple[str, str]] = set()
                retained_records: list[GeneratedStructureRecord] = []
                # Defect magnetic limits are run-global, so magnetic mode must
                # see the complete deterministic structural parent sequence.
                structural_candidates: list[Any] = []
                for task_index, result in enumerate(
                    self._execute(self._tasks(base_structures, counts), n_workers),
                    start=1,
                ):
                    self._rejected_attempts.extend(result.rejected_attempts)
                    duplicates_before = self._duplicate_count
                    batch = self._deduplicate_result(
                        result,
                        seen_structure_ids=seen_structure_ids,
                        seen_operation_ids=seen_operation_ids,
                        retained_records=retained_records,
                    )
                    if self.magnetic_generator is None:
                        self._flush(batch, output_handle=output_handle)
                    else:
                        structural_candidates.extend(batch)
                    elapsed = self._task_elapsed_seconds.pop(
                        (result.task.base_structure_id, result.task.seed),
                        0.0,
                    )
                    source = result.task.base.info.get("configurational_type", "unknown")
                    logger.info(
                        "Perturbation task %s/%s complete: source=%s base=%s accepted=%s "
                        "published=%s duplicates=%s rejected=%s elapsed=%.1fs workers=%s",
                        task_index,
                        len(base_structures),
                        source,
                        result.task.base_structure_id,
                        len(result.candidates),
                        len(batch) if self.magnetic_generator is None else 0,
                        self._duplicate_count - duplicates_before,
                        len(result.rejected_attempts),
                        elapsed,
                        n_workers,
                    )
                if self.magnetic_generator is not None:
                    magnetic_result = self.magnetic_generator.expand_structures(
                        structural_candidates
                    )
                    self._magnetic_summary = magnetic_result.summary
                    self._flush(magnetic_result.candidates, output_handle=output_handle)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            if temporary_path is None:
                raise RuntimeError("temporary generation artifact was not created")
            os.replace(temporary_path, output_file)
            temporary_path = None
            self._provenance_records = tuple(retained_records)
        except BaseException:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise
        logger.info("Saved %s structures to %s", self._total, output_file)
        return output_file

    def _deduplicate_results(
        self,
        results: Iterable[PerturbationTaskResult],
    ) -> tuple[list[Any], tuple[GeneratedStructureRecord, ...]]:
        """Return first representatives and every distinct accepted operation."""

        seen_structure_ids: set[str] = set()
        seen_operation_ids: set[tuple[str, str]] = set()
        retained_records: list[GeneratedStructureRecord] = []
        candidates: list[Any] = []
        for result in results:
            candidates.extend(
                self._deduplicate_result(
                    result,
                    seen_structure_ids=seen_structure_ids,
                    seen_operation_ids=seen_operation_ids,
                    retained_records=retained_records,
                )
            )
        return candidates, tuple(retained_records)

    def _deduplicate_result(
        self,
        result: PerturbationTaskResult,
        *,
        seen_structure_ids: set[str],
        seen_operation_ids: set[tuple[str, str]],
        retained_records: list[GeneratedStructureRecord],
    ) -> list[Any]:
        if len(result.candidates) != len(result.provenance_records):
            raise ValueError(
                "Perturbation task result must provide one provenance record per candidate"
            )
        representatives: list[Any] = []
        for candidate, record in zip(result.candidates, result.provenance_records):
            if not isinstance(record, GeneratedStructureRecord):
                raise TypeError("Perturbation task provenance must be GeneratedStructureRecord")
            structure_id = record.structure_id
            operation_key = (structure_id, record.provenance.operation_id)
            if operation_key not in seen_operation_ids:
                retained_records.append(record)
                seen_operation_ids.add(operation_key)
            if structure_id in seen_structure_ids:
                self._duplicate_count += 1
                continue
            seen_structure_ids.add(structure_id)
            representatives.append(candidate)
        return representatives

    def _execute(
        self,
        tasks: Iterable[PerturbationTask],
        n_workers: int,
    ) -> Iterator[PerturbationTaskResult]:
        """Yield worker results in task order using bounded parallel backpressure.

        Parallel mode keeps no more than twice the worker count of submitted
        futures.  Waiting for the earliest submitted future preserves canonical
        base order without an unbounded completed-result reorder buffer.
        """

        if n_workers == 1:
            for task in tasks:
                started_at = monotonic()
                result = self._execute_one(task)
                self._task_elapsed_seconds[(task.base_structure_id, task.seed)] = (
                    monotonic() - started_at
                )
                yield result
            return

        submission_window = 2 * n_workers
        task_iterator = iter(tasks)
        pending: deque[tuple[int, PerturbationTask, Any, float]] = deque()
        exhausted = False
        submitted_count = 0
        pool = ProcessPoolExecutor(max_workers=n_workers)

        def fill_submission_window() -> None:
            nonlocal exhausted, submitted_count
            while len(pending) < submission_window and not exhausted:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    return
                submitted_count += 1
                pending.append(
                    (
                        submitted_count,
                        task,
                        pool.submit(execute_perturbation_task, task),
                        monotonic(),
                    )
                )

        try:
            fill_submission_window()
            while pending:
                task_index, task, future, started_at = pending.popleft()
                while True:
                    try:
                        result = self._resolve_future(
                            task,
                            future,
                            timeout=_HEARTBEAT_INTERVAL_SECONDS,
                        )
                        break
                    except FutureTimeoutError:
                        source = task.base.info.get("configurational_type", "unknown")
                        family_plan = ", ".join(self._planned_families(task.counts, task.settings))
                        logger.info(
                            "Waiting for ordered task %s (source=%s base=%s families=%s) "
                            "after %.1fs; %s task(s) remain in the bounded window",
                            task_index,
                            source,
                            task.base_structure_id,
                            family_plan,
                            monotonic() - started_at,
                            len(pending) + 1,
                        )
                self._task_elapsed_seconds[(task.base_structure_id, task.seed)] = (
                    monotonic() - started_at
                )
                yield result
                fill_submission_window()
        except BaseException:
            for _task_index, _task, future, _started_at in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    @staticmethod
    def _execute_one(task: PerturbationTask) -> PerturbationTaskResult:
        try:
            return execute_perturbation_task(task)
        except Exception as exc:
            # Worker boundaries must preserve any task implementation failure
            # while adding task identity for the coordinator/report.
            raise PerturbationTaskError(task, exc) from exc

    @staticmethod
    def _resolve_future(
        task: PerturbationTask,
        future: Any,
        *,
        timeout: float | None = None,
    ) -> PerturbationTaskResult:
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            raise
        except Exception as exc:
            # Future implementations may wrap arbitrary worker exceptions;
            # normalize them once without treating the task as successful.
            if isinstance(exc, PerturbationTaskError):
                raise
            raise PerturbationTaskError(task, exc) from exc

    def _flush(self, candidates: Iterable[Any], *, output_handle: Any | None = None) -> None:
        batch = list(candidates)
        if not batch:
            return
        if self._output_file is None:
            raise RuntimeError("process output is not initialized")
        for candidate in batch:
            family = candidate.info.get("perturbation_type", "unknown")
            if family not in _PERTURBATION_FAMILIES:
                raise ValueError(f"Unknown perturbation type: {family!r}")
        annotate_structure_ids(batch)
        for candidate in batch:
            annotate_candidate_id(candidate)
        rendered = StringIO()
        write(rendered, batch, format="extxyz")
        if output_handle is None:
            with self._output_file.open("ab") as handle:
                handle.write(rendered.getvalue().encode("utf-8"))
        else:
            output_handle.write(rendered.getvalue().encode("utf-8"))
        for candidate in batch:
            family = candidate.info.get("perturbation_type", "unknown")
            configuration = candidate.info.get("configurational_type", "unknown")
            self._total += 1
            self._by_type[family] = self._by_type.get(family, 0) + 1
            self._by_config[configuration] = self._by_config.get(configuration, 0) + 1

    def get_provenance_records(self) -> tuple[GeneratedStructureRecord, ...]:
        """Return accepted candidate provenance in deterministic operation order."""

        return self._provenance_records

    def get_summary(self) -> dict[str, Any]:
        """Return candidate totals grouped by perturbation and configuration."""

        rejected_reason_counts: dict[str, int] = {}
        rejected_by_family: dict[str, dict[str, int]] = {}
        for rejection in self._rejected_attempts:
            rejected_reason_counts[rejection.reason] = (
                rejected_reason_counts.get(rejection.reason, 0) + 1
            )
            family_counts = rejected_by_family.setdefault(rejection.family, {})
            family_counts[rejection.reason] = family_counts.get(rejection.reason, 0) + 1
        summary = {
            "total": self._total,
            "by_type": dict(self._by_type),
            "by_config": dict(self._by_config),
            "duplicate_count": self._duplicate_count,
            "rejected_count": len(self._rejected_attempts),
            "rejected_reason_counts": rejected_reason_counts,
            "rejections_by_family": rejected_by_family,
        }
        if self._magnetic_summary is not None:
            summary["magnetic"] = self._magnetic_summary.to_dict()
        return summary


__all__ = [
    "PerturbationCoordinator",
    "PerturbationTaskError",
    "execute_perturbation_task",
    "family_applies_to_base",
]
