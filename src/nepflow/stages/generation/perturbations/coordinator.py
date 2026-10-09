"""Small typed coordinator for deterministic perturbation workers."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from io import StringIO
from multiprocessing import Manager
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
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
from nepflow.stages.generation.supercell import build_target_supercell, require_parent_topology
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
    PerturbationProgressEvent,
    PerturbationRejection,
    PerturbationSettings,
    PerturbationTask,
    PerturbationTaskResult,
)
from .provenance import annotate_generation_provenance
from .surfaces import surface_slot_count, surfaces
from .volume import volume_profile

logger = logging.getLogger(__name__)
_AUTO_WORKER_CAP = 8
_HEARTBEAT_INTERVAL_SECONDS = 30.0
_FAMILY_ORDER = (
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
)
# Publication order is intentionally base-major, then this family order, then
# ascending half-open slot windows. It is part of the extxyz/provenance contract.
_PERTURBATION_FAMILIES = frozenset(_FAMILY_ORDER)
_SLOT_BATCH_SIZE = 8


def _task_requested_units(task: PerturbationTask) -> int | None:
    """Return the number of slots represented by a bounded task, when known."""

    if task.slot_stop is None:
        return None
    return max(0, task.slot_stop - task.slot_start)


def _emit_progress(
    task: PerturbationTask,
    phase: str,
    *,
    completed_units: int | None = None,
    requested_units: int | None = None,
    detail: str | None = None,
) -> None:
    """Best-effort, non-blocking worker-to-parent progress reporting."""

    progress_queue = task.progress_queue
    if progress_queue is None:
        return
    event = PerturbationProgressEvent(
        task_key=task.progress_key,
        base_ordinal=task.base_ordinal,
        base_structure_id=task.base_structure_id,
        family=task.family or "all",
        slot_start=task.slot_start,
        slot_stop=task.slot_stop,
        phase=phase,
        completed_units=completed_units,
        requested_units=requested_units,
        timestamp=monotonic(),
        worker_pid=os.getpid(),
        detail=detail,
    )
    try:
        progress_queue.put(event, block=False)
    except (QueueFull, BrokenPipeError, EOFError, OSError):
        # Progress must never hold up a scientific worker or turn a completed
        # result into a failed result when the parent is busy draining events.
        return


class PerturbationTaskError(RuntimeError):
    """A worker failure attributed to one exact base, family, and seed."""

    def __init__(self, task: PerturbationTask, cause: BaseException) -> None:
        self.base_structure_id = task.base_structure_id
        self.seed = task.seed
        self.family = task.family
        self.slot_start = task.slot_start
        self.slot_stop = task.slot_stop
        self.cause = cause
        batch = ""
        if task.family is not None:
            batch = f" (family={task.family}, slots={task.slot_start}:{task.slot_stop})"
        super().__init__(
            f"Perturbation task failed for base={task.base_structure_id}, seed={task.seed}: "
            f"{cause}{batch}"
        )


@dataclass(slots=True)
class _TaskProgressState:
    """Parent-owned state for one currently bounded in-flight task."""

    index: int
    task: PerturbationTask
    status: str = "queued"
    submitted_at: float = 0.0
    started_at: float | None = None
    last_event_at: float | None = None
    last_phase: str = "queued"
    completed_units: int | None = None
    requested_units: int | None = None


class _ProgressTracker:
    """Aggregate bounded-window progress owned by the coordinator process."""

    def __init__(self, total: int | None, workers: int, window: int) -> None:
        self.total = total
        self.workers = workers
        self.window = window
        self.started_at = monotonic()
        self.states: dict[tuple[int, str, str, int, int | None], _TaskProgressState] = {}
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.published = 0
        self.produced = 0
        self.validated = 0
        self.accepted = 0
        self.rejected = 0
        self.duplicates = 0
        self.written = 0

    def register(self, index: int, task: PerturbationTask) -> None:
        now = monotonic()
        self.states[task.progress_key] = _TaskProgressState(
            index=index,
            task=task,
            status="submitted",
            submitted_at=now,
            requested_units=_task_requested_units(task),
        )
        self.submitted += 1

    def event(self, event: PerturbationProgressEvent) -> None:
        state = self.states.get(event.task_key)
        if state is None:
            return
        now = monotonic()
        state.last_event_at = now
        state.last_phase = event.phase
        if event.completed_units is not None:
            state.completed_units = max(state.completed_units or 0, event.completed_units)
        if event.requested_units is not None:
            state.requested_units = event.requested_units
        if event.phase == "started":
            state.status = "running"
            if state.started_at is None:
                state.started_at = now
        elif event.phase in {"generating", "validating"}:
            state.status = "running"
            if state.started_at is None:
                state.started_at = now

    def completed_result(self, task: PerturbationTask, result: PerturbationTaskResult) -> None:
        state = self.states.get(task.progress_key)
        if state is not None:
            state.status = "completed"
            state.last_event_at = monotonic()
            state.last_phase = "finished"
            state.completed_units = len(result.candidates) + len(result.rejected_attempts)
            state.requested_units = state.requested_units or _task_requested_units(task)
        self.completed += 1
        self.produced += len(result.candidates) + len(result.rejected_attempts)
        self.validated += len(result.candidates) + len(result.rejected_attempts)
        self.accepted += len(result.candidates)
        self.rejected += len(result.rejected_attempts)

    def failed_result(self, task: PerturbationTask) -> None:
        state = self.states.get(task.progress_key)
        if state is not None:
            state.status = "failed"
            state.last_event_at = monotonic()
            state.last_phase = "failed"
        self.failed += 1

    def published_result(
        self,
        task: PerturbationTask,
        *,
        written: int,
        duplicates: int,
    ) -> None:
        state = self.states.pop(task.progress_key, None)
        if state is not None:
            state.status = "published"
        self.published += 1
        self.duplicates += duplicates
        self.written += written

    def _task_label(self, task: PerturbationTask) -> str:
        stop = "?" if task.slot_stop is None else str(task.slot_stop)
        return (
            f"base={task.base_structure_id} family={task.family or 'all'} "
            f"slots={task.slot_start}:{stop}"
        )

    def snapshot(self, *, reason: str, head: _TaskProgressState | None = None) -> None:
        terminal = self.completed + self.failed
        percentage = 0.0 if not self.total else 100.0 * terminal / self.total
        active = [state for state in self.states.values() if state.status == "running"]
        waiting_for_order = sum(
            state.status == "completed" and head is not None and state.index > head.index
            for state in self.states.values()
        )
        elapsed = monotonic() - self.started_at
        total_label = "?" if self.total is None else str(self.total)
        logger.info(
            "Perturbations progress (%s): %s/%s tasks finished (%.1f%%); %s active; "
            "submitted=%s; %s in flight; %s published; %s finished awaiting order; "
            "failed=%s; elapsed=%.1fs",
            reason,
            terminal,
            total_label,
            percentage,
            len(active),
            self.submitted,
            len(self.states),
            self.published,
            waiting_for_order,
            self.failed,
            elapsed,
        )
        logger.info(
            "Candidates progress: produced=%s validated=%s accepted=%s rejected=%s "
            "duplicates=%s written=%s",
            self.produced,
            self.validated,
            self.accepted,
            self.rejected,
            self.duplicates,
            self.written,
        )
        if active:
            current = max(
                active,
                key=lambda state: monotonic() - (state.started_at or state.submitted_at),
            )
            progress = "unknown"
            if current.completed_units is not None and current.requested_units is not None:
                progress = f"{current.completed_units}/{current.requested_units}"
            last_update = (
                "never"
                if current.last_event_at is None
                else f"{monotonic() - current.last_event_at:.1f}s ago"
            )
            logger.info(
                "Active perturbation: task=%s %s phase=%s progress=%s elapsed=%.1fs "
                "last_progress=%s",
                current.index,
                self._task_label(current.task),
                current.last_phase,
                progress,
                monotonic() - (current.started_at or current.submitted_at),
                last_update,
            )
        if head is not None and head.status != "completed":
            logger.info(
                "Ordered publication blocked by task=%s (%s); %s later task(s) finished; "
                "elapsed=%.1fs",
                head.index,
                self._task_label(head.task),
                waiting_for_order,
                monotonic() - (head.started_at or head.submitted_at),
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

    _emit_progress(
        task,
        "started",
        requested_units=_task_requested_units(task),
    )
    if task.family is not None:
        return _execute_perturbation_batch(task)

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

    result = PerturbationTaskResult(
        task=task,
        candidates=tuple(accepted),
        provenance_records=tuple(provenance_records),
        rejected_attempts=tuple(rejected_attempts),
    )
    _emit_progress(
        task,
        "finished",
        completed_units=len(output),
        requested_units=_task_requested_units(task),
    )
    return result


def _execute_perturbation_batch(task: PerturbationTask) -> PerturbationTaskResult:
    """Execute one bounded family/slot window for a base structure.

    A target supercell is reconstructed once per submitted batch and shared by
    every candidate in that batch. This keeps mutable scientific state local to
    the worker and bounds memory without introducing a process-global atom
    cache. The deterministic task key, rather than completion order, controls
    all child seeds and publication order.
    """

    if task.family not in _PERTURBATION_FAMILIES:
        raise ValueError(f"unknown perturbation family: {task.family!r}")
    actual_base_id = calculate_structure_id(task.base)
    if actual_base_id != task.base_structure_id:
        raise ValueError(
            "Perturbation task base identity mismatch: "
            f"expected {task.base_structure_id}, got {actual_base_id}"
        )
    if task.slot_start < 0 or task.slot_stop is not None and task.slot_stop < task.slot_start:
        raise ValueError("perturbation batch slot window is invalid")

    settings = task.settings
    if task.prepared_supercell is None:
        # Preserve the direct/manual task compatibility path. Coordinator
        # batches always carry a parent-prepared context from _batch_tasks.
        supercell = build_target_supercell(task.base, target_n_atoms=settings.target_n_atoms)
        if supercell is None:
            raise RuntimeError("target supercell construction returned no structure")
    else:
        supercell = task.prepared_supercell.copy()
        require_parent_topology(supercell)

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

    _emit_progress(
        task,
        "generating",
        completed_units=0,
        requested_units=_task_requested_units(task),
    )
    output = _generate_family_batch(task, supercell, annotate)
    _emit_progress(
        task,
        "generating",
        completed_units=len(output),
        requested_units=_task_requested_units(task),
    )
    result = _validate_batch_candidates(task, supercell, output, candidate_records)
    _emit_progress(
        task,
        "finished",
        completed_units=len(output),
        requested_units=_task_requested_units(task),
    )
    return result


def _generate_family_batch(task: PerturbationTask, supercell: Any, annotate: Any) -> list[Any]:
    """Dispatch a bounded family window while retaining canonical family APIs."""

    family = task.family
    assert family is not None
    start = task.slot_start
    stop = task.slot_stop
    size = None if stop is None else max(0, stop - start)
    settings = task.settings
    base = task.base
    if size == 0:
        return []
    if family == "unperturbed":
        if start != 0:
            return []
        candidate = supercell.copy()
        annotate(candidate, base, family, operation_id="unperturbed")
        return [candidate]
    if family == "volume_profile":
        return volume_profile(supercell, base, settings, annotate, slot_start=start, slot_stop=stop)
    if family == "elastic_stress":
        return elastic_stress_set(
            supercell, base, settings, annotate, slot_start=start, slot_stop=stop
        )
    if family == "rattled":
        assert size is not None
        return rattled(
            supercell,
            base,
            size,
            settings,
            task.seed,
            annotate,
            slot_start=start,
            total_count=task.counts.n_rattled,
        )
    if family == "liquid":
        assert size is not None
        return liquid_snapshots(
            supercell,
            base,
            task.counts.n_liquid_configurations,
            task.counts.n_liquid_snapshots,
            settings,
            task.seed,
            annotate,
            slot_start=start,
            slot_stop=stop,
        )
    if family == "vacancy":
        assert size is not None
        return vacancies(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "interstitial":
        assert size is not None
        return interstitials(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "gas_interstitial":
        assert size is not None
        return gas_interstitials(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "substitution":
        assert size is not None
        return substitutions(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "antisite":
        assert size is not None
        return antisites(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "vacancy_interstitial":
        assert size is not None
        return vacancy_interstitial(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "gas_in_vacancy":
        assert size is not None
        return gas_in_vacancy(
            supercell, base, size, settings, None, annotate, seed=task.seed, slot_start=start
        )
    if family == "surface":
        return surfaces(
            base,
            base,
            None,
            settings,
            None,
            annotate,
            seed=task.seed,
            slot_start=start,
            slot_stop=stop,
        )
    if family == "grain_boundary":
        if start != 0:
            return []
        return grain_boundaries(
            base,
            base,
            1,
            settings,
            None,
            annotate,
            seed=task.seed,
        )
    raise AssertionError(f"unhandled perturbation family: {family}")


def _validate_batch_candidates(
    task: PerturbationTask,
    supercell: Any,
    output: list[Any],
    candidate_records: dict[int, Any],
) -> PerturbationTaskResult:
    """Validate and retain only this batch's accepted candidates."""

    _emit_progress(
        task,
        "validating",
        completed_units=0,
        requested_units=len(output),
    )
    accepted: list[Any] = []
    provenance_records: list[Any] = []
    rejected_attempts: list[PerturbationRejection] = []
    for local_index, candidate in enumerate(output):
        family = candidate.info.get("perturbation_type", "unknown")
        if family not in _PERTURBATION_FAMILIES:
            issue = CandidateValidationIssue("unknown_perturbation_family", {"family": family})
        else:
            issue = validate_generated_candidate(candidate, supercell, task.settings, family)

        record = candidate_records.get(id(candidate))
        if issue is None and record is None:
            try:
                record = annotate_generation_provenance(
                    candidate,
                    task.base,
                    family,
                    random_seed=candidate.info.get("random_seed"),
                    operation_id=(
                        f"{task.base_structure_id}:{family}:{task.slot_start + local_index}"
                    ),
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
            _emit_progress(
                task,
                "validating",
                completed_units=local_index + 1,
                requested_units=len(output),
            )
            continue

        operation_id = f"{task.base_structure_id}:{family}:{task.slot_start + local_index}"
        if record is not None:
            operation_id = str(record.provenance.operation_id)
        else:
            provenance = candidate.info.get("generation_provenance", {})
            if isinstance(provenance, dict) and provenance.get("operation_id"):
                operation_id = str(provenance["operation_id"])
        rejected_attempts.append(
            PerturbationRejection(
                parent_structure_id=task.base_structure_id,
                family=str(family),
                operation_id=operation_id,
                slot=_rejection_slot(operation_id, task.slot_start + local_index),
                reason=issue.reason,
                evidence=issue.evidence,
            )
        )
        _emit_progress(
            task,
            "validating",
            completed_units=local_index + 1,
            requested_units=len(output),
        )
    if not output:
        _emit_progress(task, "validating", completed_units=0, requested_units=0)
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
        self._task_elapsed_seconds: dict[tuple[int, str, str, int, int | None], float] = {}

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
        """Return families in the canonical publication order."""

        return tuple(
            family
            for family in _FAMILY_ORDER
            if family == "unperturbed"
            or family == "surface"
            and settings.surface_enabled
            or PerturbationCoordinator._family_slot_count(family, counts, settings) > 0
        )

    def _counts(self, **kwargs: int) -> PerturbationCounts:
        return PerturbationCounts(**kwargs)

    def _legacy_tasks(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> Iterator[PerturbationTask]:
        """Yield legacy complete-base tasks for direct callers.

        The coordinator's process paths use :meth:`_tasks`. Keeping this
        complete-task view preserves the pre-batching helper contract and the
        public ``execute_perturbation_task`` compatibility path.
        """

        task_settings = settings or self.settings
        for base_ordinal, base in enumerate(base_structures):
            yield PerturbationTask(
                base=base,
                base_structure_id=calculate_structure_id(base),
                settings=task_settings,
                counts=counts,
                seed=int(self.rng.randint(0, 2**31)),
                base_ordinal=base_ordinal,
            )

    def _tasks(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> Iterator[PerturbationTask]:
        """Yield canonical base/family/slot-window tasks for workers."""

        yield from self._batch_tasks(base_structures, counts, settings)

    @staticmethod
    def _family_slot_count(
        family: str,
        counts: PerturbationCounts,
        settings: PerturbationSettings,
        base: Any | None = None,
    ) -> int:
        """Return the deterministic number of output slots for one family."""

        if family == "unperturbed":
            return 1
        if family == "volume_profile":
            return settings.n_volume_points
        if family == "elastic_stress":
            if not settings.elastic_stress_enabled:
                return 0
            return sum(9 for value in settings.elastic_strain_amplitudes if float(value) != 0.0)
        if family == "rattled":
            return counts.n_rattled
        if family == "liquid":
            if not settings.liquid_enabled:
                return 0
            return counts.n_liquid_configurations * counts.n_liquid_snapshots
        if family == "vacancy":
            return counts.n_vacancies
        if family == "interstitial":
            return counts.n_interstitials
        if family == "gas_interstitial":
            return counts.n_gas_interstitials if settings.gas_elements else 0
        if family == "substitution":
            return counts.n_substitutions
        if family == "antisite":
            return counts.n_antisites
        if family == "vacancy_interstitial":
            return counts.n_vacancy_interstitial if settings.gas_elements else 0
        if family == "gas_in_vacancy":
            return counts.n_gas_in_vacancy if settings.gas_elements else 0
        if family == "surface":
            if not settings.surface_enabled or base is None:
                return 0
            total = surface_slot_count(base, settings)
            if len(settings.surface_miller_indices) == 1 and counts.n_surfaces > 0:
                return min(counts.n_surfaces, total)
            return total
        if family == "grain_boundary":
            return min(1, counts.n_grain_boundaries)
        raise KeyError(f"Unknown perturbation family: {family}")

    @staticmethod
    def _prepare_base_supercell(
        base: Any,
        settings: PerturbationSettings,
    ) -> Any:
        """Prepare one validated mutable-worker context for one source base."""

        prepared = build_target_supercell(base, target_n_atoms=settings.target_n_atoms)
        if prepared is None:
            raise RuntimeError("target supercell construction returned no structure")
        require_parent_topology(prepared)
        return prepared

    def _batch_tasks(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> Iterator[PerturbationTask]:
        """Yield base-major, family-major bounded slot-window tasks."""

        task_settings = settings or self.settings
        for base_ordinal, base in enumerate(base_structures):
            base_structure_id = calculate_structure_id(base)
            seed = int(self.rng.randint(0, 2**31))
            prepared_supercell = self._prepare_base_supercell(base, task_settings)
            for family in _FAMILY_ORDER:
                if family != "unperturbed" and not family_applies_to_base(
                    family, base, task_settings
                ):
                    continue
                total_slots = self._family_slot_count(family, counts, task_settings, base)
                for slot_start in range(0, total_slots, _SLOT_BATCH_SIZE):
                    yield PerturbationTask(
                        base=base,
                        base_structure_id=base_structure_id,
                        settings=task_settings,
                        counts=counts,
                        seed=seed,
                        base_ordinal=base_ordinal,
                        family=family,
                        slot_start=slot_start,
                        slot_stop=min(slot_start + _SLOT_BATCH_SIZE, total_slots),
                        prepared_supercell=prepared_supercell,
                    )

    def _batch_task_count(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> int:
        """Count bounded tasks without materialising candidate structures."""

        task_settings = settings or self.settings
        total = 0
        for base in base_structures:
            for family in _FAMILY_ORDER:
                if family != "unperturbed" and not family_applies_to_base(
                    family, base, task_settings
                ):
                    continue
                slots = self._family_slot_count(family, counts, task_settings, base)
                total += (slots + _SLOT_BATCH_SIZE - 1) // _SLOT_BATCH_SIZE
        return total

    def generate_candidates(
        self,
        base_structures: list[Any],
        *,
        counts: PerturbationCounts,
        n_workers: int = 1,
    ) -> list[Any]:
        """Generate candidates without persistence, preserving task order."""

        n_workers = self._effective_worker_count(
            n_workers,
            self._batch_task_count(base_structures, counts),
        )
        self._rejected_attempts = []
        self._provenance_records = ()
        self._duplicate_count = 0
        self._magnetic_summary = None
        self._task_elapsed_seconds = {}

        def results_with_rejections() -> Iterator[PerturbationTaskResult]:
            for result in self._execute(self._tasks(base_structures, counts), n_workers):
                self._rejected_attempts.extend(result.rejected_attempts)
                yield result
                # Candidate-only callers do not need to retain historical
                # timing entries after the result has been consumed.
                self._task_elapsed_seconds.pop(result.task.progress_key, None)

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
            # A positive legacy cap applies only to the single-orientation
            # compatibility path; multi-orientation plans remain uncapped.
            n_surfaces=n_surfaces,
            n_grain_boundaries=n_grain_boundaries,
        )
        batch_task_count = self._batch_task_count(base_structures, counts)
        n_workers = self._effective_worker_count(n_workers, batch_task_count)
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
        atom_counts = [len(base) for base in base_structures]
        atom_count_summary = "0" if not atom_counts else f"{min(atom_counts)}-{max(atom_counts)}"
        family_plan = ", ".join(self._planned_families(counts, self.settings))
        logger.info(
            "Starting perturbations: tasks=%s, workers=%s, submission_window=%s, "
            "base_atoms=%s, families=%s, output=%s",
            batch_task_count,
            n_workers,
            1 if n_workers == 1 else 2 * n_workers,
            atom_count_summary,
            family_plan,
            output_file,
        )
        progress_tracker = _ProgressTracker(
            batch_task_count,
            n_workers,
            1 if n_workers == 1 else 2 * n_workers,
        )
        progress_tracker.snapshot(reason="initial")
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
                    self._execute(
                        self._tasks(base_structures, counts),
                        n_workers,
                        total_tasks=batch_task_count,
                        progress_tracker=progress_tracker,
                    ),
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
                    duplicates = self._duplicate_count - duplicates_before
                    written = len(batch) if self.magnetic_generator is None else 0
                    progress_tracker.published_result(
                        result.task,
                        written=written,
                        duplicates=duplicates,
                    )
                    elapsed = self._task_elapsed_seconds.pop(result.task.progress_key, 0.0)
                    source = result.task.base.info.get("configurational_type", "unknown")
                    logger.info(
                        "Perturbation task %s/%s complete: source=%s base=%s accepted=%s "
                        "published=%s duplicates=%s rejected=%s elapsed=%.1fs workers=%s",
                        task_index,
                        batch_task_count,
                        source,
                        result.task.base_structure_id,
                        len(result.candidates),
                        written,
                        duplicates,
                        len(result.rejected_attempts),
                        elapsed,
                        n_workers,
                    )
                    progress_tracker.snapshot(reason="publication")
                if self.magnetic_generator is not None:
                    magnetic_result = self.magnetic_generator.expand_structures(
                        structural_candidates
                    )
                    self._magnetic_summary = magnetic_result.summary
                    self._flush(magnetic_result.candidates, output_handle=output_handle)
                    progress_tracker.written = self._total
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
        *,
        total_tasks: int | None = None,
        progress_tracker: _ProgressTracker | None = None,
    ) -> Iterator[PerturbationTaskResult]:
        """Yield worker results in task order using bounded parallel backpressure.

        Parallel mode keeps no more than twice the worker count of submitted
        futures. Every completed future is collected, while the small ready
        buffer is drained only in canonical task order.
        """

        submission_window = 1 if n_workers == 1 else 2 * n_workers
        tracker = progress_tracker or _ProgressTracker(
            total_tasks,
            n_workers,
            submission_window,
        )

        if n_workers == 1:
            progress_queue: Any = Queue(maxsize=max(32, submission_window * 4))
            for task_index, task in enumerate(tasks, start=1):
                worker_task = replace(task, progress_queue=progress_queue)
                tracker.register(task_index, task)
                started_at = monotonic()
                try:
                    result = self._execute_one(worker_task)
                except BaseException:
                    self._drain_progress_events(progress_queue, tracker)
                    tracker.failed_result(task)
                    tracker.snapshot(reason="failure")
                    raise
                self._drain_progress_events(progress_queue, tracker)
                result = replace(result, task=task)
                self._task_elapsed_seconds[task.progress_key] = monotonic() - started_at
                tracker.completed_result(task, result)
                tracker.snapshot(reason="completion")
                yield result
                # process marks publication after writing. Direct callers do
                # not, so release the bounded tracker entry on resume.
                if task.progress_key in tracker.states:
                    tracker.published_result(
                        task,
                        written=len(result.candidates),
                        duplicates=0,
                    )
            return

        task_iterator = iter(tasks)
        pending: dict[int, tuple[PerturbationTask, Any, float]] = {}
        publication_order: deque[int] = deque()
        ready: dict[int, PerturbationTaskResult] = {}
        ready_tasks: dict[int, tuple[PerturbationTask, float]] = {}
        exhausted = False
        submitted_count = 0
        manager: Any | None = None
        progress_queue: Any

        # Manager queues are pickleable under both fork and spawn. The queue
        # itself is bounded and worker writes are nonblocking, so a chatty or
        # disconnected worker cannot deadlock the process pool.
        manager = Manager()
        progress_queue = manager.Queue(maxsize=max(32, submission_window * 4))
        try:
            pool = ProcessPoolExecutor(max_workers=n_workers)
        except BaseException:
            manager.shutdown()
            raise

        def fill_submission_window() -> None:
            nonlocal exhausted, submitted_count
            while len(pending) + len(ready) < submission_window and not exhausted:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    return
                worker_task = replace(task, progress_queue=progress_queue)
                task_index = submitted_count + 1
                try:
                    future = pool.submit(execute_perturbation_task, worker_task)
                except Exception as exc:
                    raise PerturbationTaskError(task, exc) from exc
                submitted_count = task_index
                pending[task_index] = (task, future, monotonic())
                publication_order.append(task_index)
                tracker.register(task_index, task)

        def cancel_pending() -> None:
            for _task, future, _started_at in pending.values():
                future.cancel()

        def publish_ready() -> Iterator[PerturbationTaskResult]:
            while publication_order and publication_order[0] in ready:
                task_index = publication_order.popleft()
                task, _started_at = ready_tasks.pop(task_index)
                result = ready.pop(task_index)
                yield result
                # If the consumer is process(), it has marked the result as
                # published while this generator was suspended. Other
                # consumers receive the same bounded-window accounting.
                if task.progress_key in tracker.states:
                    tracker.published_result(
                        task,
                        written=len(result.candidates),
                        duplicates=0,
                    )
                fill_submission_window()

        try:
            fill_submission_window()
            while pending or ready:
                self._drain_progress_events(progress_queue, tracker)

                # The synchronous harness used by existing unit tests
                # predates concurrent.futures.Future and intentionally only
                # resolves work when result() is consumed. Preserve that
                # compatibility while real futures use FIRST_COMPLETED below.
                supports_completion_wait = all(
                    callable(getattr(future, "done", None))
                    for _task, future, _started_at in pending.values()
                )
                if pending and not supports_completion_wait:
                    task_index = publication_order[0]
                    task, future, started_at = pending.pop(task_index)
                    try:
                        result = self._resolve_future(
                            task,
                            future,
                            timeout=_HEARTBEAT_INTERVAL_SECONDS,
                        )
                    except FutureTimeoutError:
                        pending[task_index] = (task, future, started_at)
                        self._drain_progress_events(progress_queue, tracker)
                        head = tracker.states.get(task.progress_key)
                        source = task.base.info.get("configurational_type", "unknown")
                        stop = "?" if task.slot_stop is None else str(task.slot_stop)
                        logger.info(
                            "Waiting for ordered task %s (source=%s base=%s family=%s "
                            "slots=%s:%s) after %.1fs; %s task(s) remain in the bounded window",
                            task_index,
                            source,
                            task.base_structure_id,
                            task.family or "all",
                            task.slot_start,
                            stop,
                            monotonic() - started_at,
                            len(pending) + len(ready),
                        )
                        tracker.snapshot(reason="heartbeat", head=head)
                        continue
                    except BaseException:
                        tracker.failed_result(task)
                        tracker.snapshot(reason="failure")
                        raise
                    ready_tasks[task_index] = (task, started_at)
                    result = replace(result, task=task)
                    self._task_elapsed_seconds[task.progress_key] = monotonic() - started_at
                    tracker.completed_result(task, result)
                    tracker.snapshot(
                        reason="completion", head=tracker.states.get(task.progress_key)
                    )
                    ready[task_index] = result
                    yield from publish_ready()
                    continue

                if pending:
                    futures = [future for _task, future, _started_at in pending.values()]
                    try:
                        done, _not_done = wait(
                            futures,
                            timeout=_HEARTBEAT_INTERVAL_SECONDS,
                            return_when=FIRST_COMPLETED,
                        )
                    except Exception as exc:
                        head_entry = pending.get(publication_order[0])
                        if head_entry is not None:
                            raise PerturbationTaskError(head_entry[0], exc) from exc
                        raise
                    self._drain_progress_events(progress_queue, tracker)
                    if not done:
                        head_task = pending.get(publication_order[0])
                        head = (
                            None
                            if head_task is None
                            else tracker.states.get(head_task[0].progress_key)
                        )
                        tracker.snapshot(reason="heartbeat", head=head)
                        continue
                    future_to_index = {
                        future: task_index
                        for task_index, (_task, future, _started_at) in pending.items()
                    }
                    for future in done:
                        task_index = future_to_index[future]
                        task, _future, started_at = pending.pop(task_index)
                        try:
                            result = self._resolve_future(task, future)
                        except BaseException:
                            tracker.failed_result(task)
                            tracker.snapshot(reason="failure")
                            raise
                        result = replace(result, task=task)
                        ready_tasks[task_index] = (task, started_at)
                        self._task_elapsed_seconds[task.progress_key] = monotonic() - started_at
                        tracker.completed_result(task, result)
                        ready[task_index] = result
                        self._drain_progress_events(progress_queue, tracker)
                        head_state = None
                        if publication_order:
                            head_entry = pending.get(publication_order[0])
                            if head_entry is not None:
                                head_state = tracker.states.get(head_entry[0].progress_key)
                        tracker.snapshot(reason="completion", head=head_state)
                    yield from publish_ready()
        except BaseException:
            cancel_pending()
            self._shutdown_pool(pool, terminate_running=True)
            raise
        else:
            self._shutdown_pool(pool, terminate_running=False)
        finally:
            if manager is not None:
                manager.shutdown()

    @staticmethod
    def _drain_progress_events(progress_queue: Any, tracker: _ProgressTracker) -> None:
        """Drain only currently available bounded progress notifications."""

        while True:
            try:
                event = progress_queue.get_nowait()
            except (QueueEmpty, EOFError, OSError):
                return
            if isinstance(event, PerturbationProgressEvent):
                tracker.event(event)

    @staticmethod
    def _shutdown_pool(pool: Any, *, terminate_running: bool) -> None:
        """Shut down a process pool, terminating active workers on failure.

        Python 3.11--3.13 do not expose the public worker-termination methods
        added later.  The private process handles are the only compatible
        fallback; they are used only during failure/interruption cleanup.
        """

        if not terminate_running:
            pool.shutdown(wait=True, cancel_futures=False)
            return

        terminate_workers = getattr(pool, "terminate_workers", None)
        if callable(terminate_workers):
            terminate_workers()
            return

        processes = getattr(pool, "_processes", None)
        if not isinstance(processes, dict):
            # Test doubles and alternate executors may not expose process
            # handles; retain their normal cancellation contract.
            pool.shutdown(wait=False, cancel_futures=True)
            return

        workers = list(processes.values())
        for worker in workers:
            try:
                if worker.is_alive():
                    worker.terminate()
            except (OSError, AttributeError):
                continue
        for worker in workers:
            try:
                worker.join(timeout=5.0)
            except (OSError, AttributeError):
                continue
        for worker in workers:
            try:
                if worker.is_alive():
                    kill = getattr(worker, "kill", None)
                    if callable(kill):
                        kill()
                    worker.join(timeout=5.0)
            except (OSError, AttributeError):
                continue
        pool.shutdown(wait=True, cancel_futures=True)

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
