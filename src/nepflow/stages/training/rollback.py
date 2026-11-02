"""Operator-requested rollback cleanup for authoritative NEP training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from nepflow.errors import StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.scheduler import Scheduler
from nepflow.state import StateStore


@dataclass(frozen=True, slots=True)
class TrainingRollbackResult:
    """Scheduler and ledger evidence collected before stage rewind."""

    cancelled_attempt_ids: tuple[str, ...]
    cancelled_job_ids: tuple[str, ...]
    invalidated_campaign_ids: tuple[str, ...]
    execution_generations: dict[str, int]
    next_execution_generations: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancelled_attempt_ids": self.cancelled_attempt_ids,
            "cancelled_job_ids": self.cancelled_job_ids,
            "invalidated_campaign_ids": self.invalidated_campaign_ids,
            "execution_generations": self.execution_generations,
            "next_execution_generations": self.next_execution_generations,
            "training_authority_invalidated": True,
        }


class TrainingRollbackService:
    """Cancel current NEP work and append an explicit invalidation boundary."""

    def __init__(self, scheduler: Scheduler, *, sleeper: Any = time.sleep) -> None:
        self.scheduler = scheduler
        self._sleep = sleeper

    def rollback(
        self,
        *,
        project_id: str,
        state_store: StateStore,
        scheduler: Scheduler | None = None,
        poll_interval_seconds: int = 0,
    ) -> TrainingRollbackResult:
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")
        active_scheduler = scheduler or self.scheduler
        authorities = state_store.list_authoritative_training_campaigns(project_id)
        live: list[tuple[dict[str, Any], dict[str, Any], SchedulerJobState]] = []
        for authority in authorities:
            generation = int(authority["execution_generation"])
            campaign_id = str(authority["campaign_id"])
            for attempt in state_store.list_training_attempt_projections(
                campaign_id, generation
            ):
                if attempt.get("status") not in {"submitting", "submitted", "running"}:
                    continue
                job_id = attempt.get("job_id")
                if not isinstance(job_id, str) or not job_id.strip():
                    raise StateError(
                        "Cannot rollback training; authoritative live attempt has no job ID: "
                        f"{attempt.get('attempt_id')}"
                    )
                state = self._cancel_and_confirm(
                    job_id,
                    active_scheduler,
                    poll_interval_seconds,
                )
                live.append((authority, attempt, state))

        cancelled_attempt_ids = tuple(str(attempt["attempt_id"]) for _, attempt, _ in live)
        cancelled_job_ids = tuple(str(attempt["job_id"]) for _, attempt, _ in live)
        invalidated_campaign_ids: list[str] = []
        generations: dict[str, int] = {}
        next_generations: dict[str, int] = {}

        for authority in authorities:
            campaign_id = str(authority["campaign_id"])
            generation = int(authority["execution_generation"])
            generations[campaign_id] = generation
            next_generations[campaign_id] = generation + 1
            invalidated_campaign_ids.append(campaign_id)

        for authority, attempt, state in live:
            campaign_id = str(authority["campaign_id"])
            payload = dict(attempt)
            payload.update(
                {
                    "campaign_id": campaign_id,
                    "status": "cancelled",
                    "scheduler_state": state.value,
                    "failure_reason": "explicit_workflow_rollback",
                    "execution_generation": int(authority["execution_generation"]),
                }
            )
            self._append_event(
                state_store,
                "training_attempt",
                campaign_id,
                "rollback_cancelled",
                payload,
            )

        for authority in authorities:
            campaign_id = str(authority["campaign_id"])
            generation = int(authority["execution_generation"])
            candidates = state_store.list_training_candidate_projections(campaign_id, generation)
            for candidate in candidates:
                if candidate.get("status") in {"completed", "failed", "invalidated"}:
                    candidate_status = candidate.get("status")
                    if candidate_status == "completed":
                        candidate_status = "invalidated"
                    else:
                        continue
                else:
                    candidate_status = "invalidated"
                payload = dict(candidate)
                payload.update(
                    {
                        "campaign_id": campaign_id,
                        "status": candidate_status,
                        "job_id": None,
                        "failure_reason": "explicit_workflow_rollback",
                        "execution_generation": generation,
                    }
                )
                self._append_event(
                    state_store,
                    "training_candidate",
                    campaign_id,
                    "rollback_invalidated",
                    payload,
                )
            campaign_payload = self._latest_campaign_payload(
                state_store, campaign_id, generation
            )
            campaign_payload.update(
                {
                    "campaign_id": campaign_id,
                    "status": "invalidated",
                    "execution_generation": generation,
                    "rollback_source": "explicit_workflow_rollback",
                }
            )
            self._append_event(
                state_store,
                "training_campaign",
                campaign_id,
                "rollback_invalidated",
                campaign_payload,
            )

        next_generations = state_store.invalidate_training_execution_authority(
            project_id,
            source="explicit_workflow_rollback",
        )

        return TrainingRollbackResult(
            cancelled_attempt_ids=cancelled_attempt_ids,
            cancelled_job_ids=cancelled_job_ids,
            invalidated_campaign_ids=tuple(invalidated_campaign_ids),
            execution_generations=generations,
            next_execution_generations=next_generations,
        )

    def _cancel_and_confirm(
        self,
        job_id: str,
        scheduler: Scheduler,
        poll_interval_seconds: int,
    ) -> SchedulerJobState:
        reconciliation = scheduler.reconcile(job_id)
        if reconciliation.job.state is SchedulerJobState.UNKNOWN:
            raise StateError(f"Cannot determine scheduler state for training job {job_id}")
        if reconciliation.job.state.is_active:
            scheduler.cancel(job_id)
            for _ in range(60):
                reconciliation = scheduler.reconcile(job_id)
                state = reconciliation.job.state
                if state is SchedulerJobState.UNKNOWN:
                    raise StateError(
                        f"Cannot confirm cancellation of training job {job_id}: "
                        "scheduler state unknown"
                    )
                if state is SchedulerJobState.NOT_FOUND:
                    raise StateError(
                        f"Cannot confirm cancellation of training job {job_id}: job not found"
                    )
                if not state.is_active:
                    return state
                if poll_interval_seconds:
                    self._sleep(poll_interval_seconds)
            raise StateError(f"Training job {job_id} remains active after cancellation")
        if reconciliation.job.state is SchedulerJobState.NOT_FOUND:
            raise StateError(f"Cannot establish terminal state for training job {job_id}")
        return reconciliation.job.state

    @staticmethod
    def _latest_campaign_payload(
        state_store: StateStore,
        campaign_id: str,
        generation: int,
    ) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for event in state_store.list_training_events("training_campaign", campaign_id):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if int(payload.get("execution_generation", 1)) == generation:
                latest = dict(payload)
        return latest

    @staticmethod
    def _append_event(
        state_store: StateStore,
        entity_type: str,
        campaign_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        state_store.record_training_event(
            entity_type,
            campaign_id,
            event_type,
            payload,
            event_id=f"training:{entity_type}:{campaign_id}:rollback:{uuid4().hex}",
        )


__all__ = ["TrainingRollbackResult", "TrainingRollbackService"]
