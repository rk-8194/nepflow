"""Thin canonical validation service over explicit cases and MLIP protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.simulation import StaticPrediction, StaticPredictionBackend

from .preparation import prepare_validation_cases
from .protocols import ValidationPreparation
from .resolution import resolve_model_dataset


@dataclass(frozen=True, slots=True)
class ValidationStage:
    """Prepare and parse one explicit model/dataset validation run.

    Scheduler orchestration remains outside this boundary.  ``parse_completed``
    consumes only the output in each identity-bound case directory and routes
    it through the existing :class:`StaticPredictionBackend` protocol.
    """

    project_dir: Path
    model_run_id: str
    dataset_id: str | None = None
    state_store: Any | None = None
    backend: StaticPredictionBackend | None = None

    def prepare(self) -> ValidationPreparation:
        return prepare_validation_cases(
            self.project_dir,
            self.model_run_id,
            dataset_id=self.dataset_id,
            state_store=self.state_store,
            backend=self.backend if isinstance(self.backend, GpumdBackend) else None,
        )

    def parse_completed(self, preparation: ValidationPreparation) -> tuple[StaticPrediction, ...]:
        resolved = resolve_model_dataset(
            self.project_dir,
            preparation.model_run_id,
            dataset_id=preparation.dataset_id,
            state_store=self.state_store,
        )
        backend = self.backend or GpumdBackend()
        predictions: list[StaticPrediction] = []
        for case in preparation.cases:
            request = case.static_prediction_request(resolved.model_run)
            predictions.append(backend.predict(request))
        return tuple(predictions)


__all__ = ["ValidationStage"]
