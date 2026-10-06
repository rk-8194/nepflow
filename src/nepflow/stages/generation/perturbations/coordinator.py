"""Small typed coordinator for deterministic perturbation workers."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write

from nepflow.config.models import ALL_SOURCES
from nepflow.domain.identities import annotate_structure_ids, calculate_structure_id
from nepflow.io.atomic import atomic_write_bytes, atomic_write_text
from nepflow.stages.generation.supercell import build_target_supercell
from nepflow.stages.generation.validation import (
    CandidateValidationIssue,
    validate_generated_candidate,
)

from .defects import (
    gas_in_vacancy,
    gas_interstitials,
    interstitials,
    vacancies,
    vacancy_interstitial,
)
from .displacements import rattled
from .elastic import elastic_stress_set
from .liquid import liquid_snapshots
from .models import (
    PerturbationCounts,
    PerturbationRejection,
    PerturbationSettings,
    PerturbationTask,
    PerturbationTaskResult,
)
from .provenance import annotate_generation_provenance
from .volume import volume_profile

logger = logging.getLogger(__name__)
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
        "vacancy_interstitial",
        "gas_in_vacancy",
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

    def __init__(self, settings: PerturbationSettings | None = None, **kwargs: Any) -> None:
        if settings is not None and kwargs:
            raise TypeError("pass either typed settings or legacy keyword settings")
        self.settings = settings or PerturbationSettings(**kwargs)
        self.rng = np.random.RandomState(self.settings.random_seed)
        self._total = 0
        self._by_type: dict[str, int] = {}
        self._by_config: dict[str, int] = {}
        self._rejected_attempts: list[PerturbationRejection] = []
        self._output_file: Path | None = None

    def _counts(self, **kwargs: int) -> PerturbationCounts:
        return PerturbationCounts(**kwargs)

    def _tasks(
        self,
        base_structures: Iterable[Any],
        counts: PerturbationCounts,
        settings: PerturbationSettings | None = None,
    ) -> list[PerturbationTask]:
        bases = list(base_structures)
        seeds = self.rng.randint(0, 2**31, size=len(bases)).tolist()
        task_settings = settings or self.settings
        return [
            PerturbationTask(
                base=base,
                base_structure_id=calculate_structure_id(base),
                settings=task_settings,
                counts=counts,
                seed=int(seeds[index]),
            )
            for index, base in enumerate(bases)
        ]

    def generate_candidates(
        self,
        base_structures: list[Any],
        *,
        counts: PerturbationCounts,
        n_workers: int = 1,
    ) -> list[Any]:
        """Generate candidates without persistence, preserving task order."""

        if n_workers == 0:
            n_workers = max(1, os.cpu_count() or 1)
        if n_workers < 1:
            raise ValueError("n_workers must be non-negative or at least one")
        self._rejected_attempts = []
        results = self._execute(self._tasks(base_structures, counts), n_workers)
        self._rejected_attempts.extend(
            rejection for result in results for rejection in result.rejected_attempts
        )
        return [candidate for result in results for candidate in result.candidates]

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
        n_vacancy_interstitial: int = 0,
        n_gas_in_vacancy: int = 0,
        n_workers: int = 0,
    ) -> Path:
        """Run typed tasks and persist candidates in deterministic base order."""

        if n_workers == 0:
            n_workers = max(1, os.cpu_count() or 1)
        if n_workers < 1:
            raise ValueError("n_workers must be non-negative or at least one")
        self._total = 0
        self._by_type = {}
        self._by_config = {}
        self._rejected_attempts = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file = output_dir / "generated_structures.xyz"
        atomic_write_text(self._output_file, "", encoding="utf-8")
        counts = self._counts(
            n_rattled=n_rattled,
            n_liquid_configurations=n_liquid_configurations,
            n_liquid_snapshots=n_liquid_snapshots,
            n_vacancies=n_vacancies,
            n_interstitials=n_interstitials,
            n_gas_interstitials=n_gas_interstitials,
            n_vacancy_interstitial=n_vacancy_interstitial,
            n_gas_in_vacancy=n_gas_in_vacancy,
        )
        for result in self._execute(self._tasks(base_structures, counts), n_workers):
            self._rejected_attempts.extend(result.rejected_attempts)
            self._flush(result.candidates)
        logger.info("Saved %s structures to %s", self._total, self._output_file)
        return self._output_file

    def _execute(
        self,
        tasks: list[PerturbationTask],
        n_workers: int,
    ) -> list[PerturbationTaskResult]:
        if n_workers == 1:
            return [self._execute_one(task) for task in tasks]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(execute_perturbation_task, task) for task in tasks]
            return [self._resolve_future(task, future) for task, future in zip(tasks, futures)]

    @staticmethod
    def _execute_one(task: PerturbationTask) -> PerturbationTaskResult:
        try:
            return execute_perturbation_task(task)
        except Exception as exc:
            # Worker boundaries must preserve any task implementation failure
            # while adding task identity for the coordinator/report.
            raise PerturbationTaskError(task, exc) from exc

    @staticmethod
    def _resolve_future(task: PerturbationTask, future: Any) -> PerturbationTaskResult:
        try:
            return future.result()
        except Exception as exc:
            # Future implementations may wrap arbitrary worker exceptions;
            # normalize them once without treating the task as successful.
            if isinstance(exc, PerturbationTaskError):
                raise
            raise PerturbationTaskError(task, exc) from exc

    def _flush(self, candidates: Iterable[Any]) -> None:
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
        rendered = StringIO()
        write(rendered, batch, format="extxyz")
        existing = self._output_file.read_bytes()
        atomic_write_bytes(self._output_file, existing + rendered.getvalue().encode("utf-8"))
        for candidate in batch:
            family = candidate.info.get("perturbation_type", "unknown")
            configuration = candidate.info.get("configurational_type", "unknown")
            self._total += 1
            self._by_type[family] = self._by_type.get(family, 0) + 1
            self._by_config[configuration] = self._by_config.get(configuration, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        """Return candidate totals grouped by perturbation and configuration."""

        rejected_reason_counts: dict[str, int] = {}
        for rejection in self._rejected_attempts:
            rejected_reason_counts[rejection.reason] = (
                rejected_reason_counts.get(rejection.reason, 0) + 1
            )
        return {
            "total": self._total,
            "by_type": dict(self._by_type),
            "by_config": dict(self._by_config),
            "rejected_count": len(self._rejected_attempts),
            "rejected_reason_counts": rejected_reason_counts,
        }


__all__ = [
    "PerturbationCoordinator",
    "PerturbationTaskError",
    "execute_perturbation_task",
    "family_applies_to_base",
]
