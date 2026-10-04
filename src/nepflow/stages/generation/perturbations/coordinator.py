"""Small typed coordinator for deterministic perturbation workers."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase.io import write

from nepflow.domain.identities import annotate_structure_ids, calculate_structure_id
from nepflow.stages.generation.supercell import build_target_supercell

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

    rng = np.random.RandomState(task.seed)
    provenance_records: list[Any] = []

    def annotate(
        candidate: Any,
        base: Any,
        family: str,
        *,
        random_seed: int | None = None,
        parameters: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> Any:
        stochastic = {
            "rattled",
            "liquid",
            "vacancy",
            "interstitial",
            "gas_interstitial",
            "vacancy_interstitial",
            "gas_in_vacancy",
        }
        record = annotate_generation_provenance(
            candidate,
            base,
            family,
            random_seed=(task.seed if family in stochastic else random_seed),
            parameters=parameters,
            operation_id=(
                f"{task.base_structure_id}:{operation_id}" if operation_id is not None else None
            ),
        )
        provenance_records.append(record)
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
    output.extend(volume_profile(supercell, task.base, settings, annotate))
    output.extend(elastic_stress_set(supercell, task.base, settings, annotate))
    output.extend(
        rattled(supercell, task.base, task.counts.n_rattled, settings, task.seed, annotate)
    )
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
    output.extend(
        vacancies(
            supercell,
            task.base,
            task.counts.n_vacancies,
            settings,
            rng,
            annotate,
            seed=task.seed,
        )
    )
    output.extend(
        interstitials(
            supercell,
            task.base,
            task.counts.n_interstitials,
            settings,
            rng,
            annotate,
            seed=task.seed,
        )
    )
    if settings.gas_elements:
        output.extend(
            gas_interstitials(
                supercell,
                task.base,
                task.counts.n_gas_interstitials,
                settings,
                rng,
                annotate,
                seed=task.seed,
            )
        )
        output.extend(
            vacancy_interstitial(
                supercell,
                task.base,
                task.counts.n_vacancy_interstitial,
                settings,
                rng,
                annotate,
                seed=task.seed,
            )
        )
        output.extend(
            gas_in_vacancy(
                supercell,
                task.base,
                task.counts.n_gas_in_vacancy,
                settings,
                rng,
                annotate,
                seed=task.seed,
            )
        )
    return PerturbationTaskResult(
        task=task,
        candidates=tuple(output),
        provenance_records=tuple(provenance_records),
    )


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
        self._output_file: Path | None = None

    def _counts(self, **kwargs: int) -> PerturbationCounts:
        return PerturbationCounts(**kwargs)

    def _tasks(
        self, base_structures: Iterable[Any], counts: PerturbationCounts
    ) -> list[PerturbationTask]:
        bases = list(base_structures)
        seeds = self.rng.randint(0, 2**31, size=len(bases)).tolist()
        return [
            PerturbationTask(
                base=base,
                base_structure_id=calculate_structure_id(base),
                settings=self.settings,
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
        return [
            candidate
            for result in self._execute(self._tasks(base_structures, counts), n_workers)
            for candidate in result.candidates
        ]

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file = output_dir / "generated_structures.xyz"
        self._output_file.write_text("")
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
            raise PerturbationTaskError(task, exc) from exc

    @staticmethod
    def _resolve_future(task: PerturbationTask, future: Any) -> PerturbationTaskResult:
        try:
            return future.result()
        except Exception as exc:
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
        write(str(self._output_file), batch, append=True)
        for candidate in batch:
            family = candidate.info.get("perturbation_type", "unknown")
            configuration = candidate.info.get("configurational_type", "unknown")
            self._total += 1
            self._by_type[family] = self._by_type.get(family, 0) + 1
            self._by_config[configuration] = self._by_config.get(configuration, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        return {
            "total": self._total,
            "by_type": dict(self._by_type),
            "by_config": dict(self._by_config),
        }


__all__ = [
    "PerturbationCoordinator",
    "PerturbationTaskError",
    "execute_perturbation_task",
]
