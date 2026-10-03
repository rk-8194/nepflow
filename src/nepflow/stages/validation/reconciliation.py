"""StateStore-backed reconciliation of prepared GPUMD validation cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from nepflow.domain.identities import ValidationRunIdentity
from nepflow.domain.models import ModelRunRecord, ValidationRunRecord
from nepflow.errors import StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.resources import JobResources
from nepflow.io.json import to_jsonable
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.simulation import StaticPrediction

from .protocols import ValidationCaseSpec


_ACTIVE = frozenset({"submitted", "running"})
_TERMINAL = frozenset({"completed", "failed"})


@dataclass(frozen=True, slots=True)
class ValidationExecutionRecord:
    """The latest persisted attempt for one identity-bound validation case."""

    validation_run_id: str
    case: ValidationCaseSpec
    attempt_id: str
    attempt_number: int
    status: str = "pending"
    job_id: str | None = None
    job_name: str | None = None
    scheduler_state: str | None = None
    failure_reason: str | None = None
    prediction: StaticPrediction | None = None

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def output_path(self) -> Path:
        return self.case.output_path

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL


# These concise names make the case/attempt boundary discoverable without
# duplicating a second mutable runtime model.
ValidationCaseExecution = ValidationExecutionRecord
ValidationAttempt = ValidationExecutionRecord


@dataclass(frozen=True, slots=True)
class ValidationReconciliationResult:
    """The state table after one idempotent reconciliation pass."""

    validation_run_id: str
    status: str
    cases: tuple[ValidationExecutionRecord, ...]

    @property
    def records(self) -> tuple[ValidationExecutionRecord, ...]:
        """Compatibility spelling shared with the DFT reconciler."""

        return self.cases

    @property
    def all_terminal(self) -> bool:
        return bool(self.cases) and all(case.terminal for case in self.cases)

    @property
    def all_successful(self) -> bool:
        return bool(self.cases) and all(case.status == "completed" for case in self.cases)

    @property
    def any_failed(self) -> bool:
        return any(case.status == "failed" for case in self.cases)

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for case in self.cases:
            result[case.status] = result.get(case.status, 0) + 1
        return result


class ValidationReconciliationOrchestrator:
    """Advance prepared validation cases using scheduler and backend evidence.

    Case definitions are immutable snapshots in StateStore.  Only the latest
    case status is projected from append-only ``validation_case`` events;
    every attempt remains available through ``validation_attempt`` events.
    """

    def __init__(
        self,
        *,
        validation_run: ValidationRunIdentity,
        model: ModelRunRecord,
        cases: Sequence[ValidationCaseSpec],
        state_store: Any,
        scheduler: Any,
        backend: Any | None = None,
        max_concurrent: int = 1,
        max_attempts: int = 1,
        resources: JobResources | None = None,
        job_name_prefix: str | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not cases:
            raise ValueError("validation reconciliation requires at least one case")
        if validation_run.model_run_id != model.model_run_id:
            raise StateError("validation run model identity does not match the model record")
        if validation_run.dataset_id != model.identity.dataset_id:
            raise StateError("validation run dataset identity does not match the model record")
        case_ids = [case.case_id for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise StateError("validation cases must have unique case IDs")
        ordinals = [case.ordinal for case in cases]
        if len(set(ordinals)) != len(ordinals):
            raise StateError("validation cases must have unique ordinals")
        for case in cases:
            if case.model_run_id != model.model_run_id:
                raise StateError(f"validation case model identity does not match: {case.case_id}")
            if case.dataset_id != validation_run.dataset_id:
                raise StateError(f"validation case dataset identity does not match: {case.case_id}")
        if not callable(getattr(state_store, "append_event", None)) and not callable(
            getattr(state_store, "record_validation_event", None)
        ):
            raise TypeError("ValidationReconciliationOrchestrator requires an event-capable StateStore")

        self.validation_run = validation_run
        self.model = model
        self._cases = tuple(sorted(cases, key=lambda item: (item.ordinal, item.case_id)))
        self.state_store = state_store
        self.scheduler = scheduler
        self.backend = backend or GpumdBackend()
        self.max_concurrent = max_concurrent
        self.max_attempts = max_attempts
        self.resources = resources
        self.job_name_prefix = job_name_prefix
        self._ensure_run()
        self._ensure_cases()

    @property
    def validation_run_id(self) -> str:
        return self.validation_run.validation_run_id

    def _events(self, entity_type: str, entity_id: str | None = None) -> list[dict[str, Any]]:
        lister = getattr(self.state_store, "list_validation_events", None)
        if callable(lister):
            return list(lister(entity_type, entity_id))
        return list(self.state_store.list_events(entity_type=entity_type, entity_id=entity_id))

    def _append(
        self,
        entity_type: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        events = self._events(entity_type, self.validation_run_id)
        event_id = f"validation:{entity_type}:{self.validation_run_id}:{len(events):012d}"
        occurred_at = datetime.now(timezone.utc)
        existing_times = [
            event.get("occurred_at")
            for event in events
            if isinstance(event.get("occurred_at"), str)
        ]
        if existing_times:
            try:
                latest = max(datetime.fromisoformat(value) for value in existing_times)
                if occurred_at <= latest:
                    occurred_at = latest + timedelta(microseconds=1)
            except ValueError:
                pass
        recorder = getattr(self.state_store, "record_validation_event", None)
        kwargs = {
            "event_id": event_id,
            "occurred_at": occurred_at.isoformat(),
        }
        if callable(recorder):
            try:
                recorder(
                    entity_type,
                    self.validation_run_id,
                    event_type,
                    to_jsonable(dict(payload)),
                    **kwargs,
                )
            except TypeError:
                recorder(entity_type, self.validation_run_id, event_type, to_jsonable(dict(payload)))
            return
        self.state_store.append_event(
            event_id,
            entity_type,
            self.validation_run_id,
            event_type,
            to_jsonable(dict(payload)),
            occurred_at=kwargs["occurred_at"],
        )

    def _ensure_run(self) -> None:
        getter = getattr(self.state_store, "get_validation_run", None)
        existing = getter(self.validation_run_id) if callable(getter) else None
        if existing is not None:
            persisted = existing.get("identity", existing.get("identity_json", {}))
            if isinstance(persisted, Mapping) and dict(persisted) != self.validation_run.to_dict():
                raise StateError(f"Validation-run identity conflict: {self.validation_run_id}")
            return
        upsert = getattr(self.state_store, "upsert_validation_run", None)
        if callable(upsert):
            upsert(
                ValidationRunRecord(
                    identity=self.validation_run,
                    metadata={"case_ids": [case.case_id for case in self._cases]},
                ),
                status="pending",
            )

    def _ensure_cases(self) -> None:
        existing = self._latest_case_payloads()
        for case in self._cases:
            payload = existing.get(case.case_id)
            if payload is not None:
                persisted_case = payload.get("case")
                if persisted_case != case.to_dict():
                    raise StateError(f"Validation case identity conflict: {case.case_id}")
                continue
            record = ValidationExecutionRecord(
                validation_run_id=self.validation_run_id,
                case=case,
                attempt_id=self._attempt_id(case.case_id, 1),
                attempt_number=1,
            )
            self._append_record(record, event_type="prepared")

    def _latest_case_payloads(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._events("validation_case", self.validation_run_id):
            payload = event.get("payload", {})
            case_id = payload.get("case_id")
            if case_id:
                latest[str(case_id)] = dict(payload)
        return latest

    def _attempt_payloads(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._events("validation_attempt", self.validation_run_id):
            payload = event.get("payload", {})
            attempt_id = payload.get("attempt_id")
            if attempt_id:
                latest[str(attempt_id)] = dict(payload)
        return latest

    @staticmethod
    def _attempt_id(case_id: str, attempt_number: int) -> str:
        return f"{case_id}:attempt:{attempt_number}"

    def _record_from_payload(self, payload: Mapping[str, Any]) -> ValidationExecutionRecord:
        case = ValidationCaseSpec.from_mapping(payload["case"])
        return ValidationExecutionRecord(
            validation_run_id=self.validation_run_id,
            case=case,
            attempt_id=str(payload["attempt_id"]),
            attempt_number=int(payload["attempt_number"]),
            status=str(payload["status"]),
            job_id=payload.get("job_id"),
            job_name=payload.get("job_name"),
            scheduler_state=payload.get("scheduler_state"),
            failure_reason=payload.get("failure_reason"),
        )

    def cases(self) -> tuple[ValidationExecutionRecord, ...]:
        payloads = self._latest_case_payloads()
        records = [self._record_from_payload(payloads[case.case_id]) for case in self._cases]
        return tuple(records)

    def attempts(self, case_id: str | None = None) -> tuple[ValidationExecutionRecord, ...]:
        records = []
        for payload in self._attempt_payloads().values():
            if case_id is not None and payload.get("case_id") != case_id:
                continue
            records.append(self._record_from_payload(payload))
        return tuple(sorted(records, key=lambda item: (item.case.ordinal, item.attempt_number)))

    def _payload(self, record: ValidationExecutionRecord) -> dict[str, Any]:
        return {
            "validation_run_id": self.validation_run_id,
            "case_id": record.case.case_id,
            "case": record.case.to_dict(),
            "attempt_id": record.attempt_id,
            "attempt_number": record.attempt_number,
            "status": record.status,
            "job_id": record.job_id,
            "job_name": record.job_name,
            "scheduler_state": record.scheduler_state,
            "failure_reason": record.failure_reason,
            "output_path": str(record.case.output_path),
        }

    def _append_record(self, record: ValidationExecutionRecord, *, event_type: str) -> None:
        payload = self._payload(record)
        self._append("validation_case", event_type, payload)
        self._append("validation_attempt", event_type, payload)

    def _save_run_status(self, status: str) -> None:
        upsert = getattr(self.state_store, "upsert_validation_run", None)
        if not callable(upsert):
            return
        completed_at = (
            datetime.now(timezone.utc).isoformat()
            if status in _TERMINAL
            else None
        )
        upsert(
            ValidationRunRecord(
                identity=self.validation_run,
                metadata={"case_ids": [case.case_id for case in self._cases]},
            ),
            status=status,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=completed_at,
        )

    def _save_status(
        self,
        record: ValidationExecutionRecord,
        status: str,
        *,
        scheduler_state: str | None = None,
        job_id: str | None = None,
        job_name: str | None = None,
        failure_reason: str | None = None,
        prediction: StaticPrediction | None = None,
    ) -> ValidationExecutionRecord:
        updated = replace(
            record,
            status=status,
            scheduler_state=scheduler_state if scheduler_state is not None else record.scheduler_state,
            job_id=record.job_id if job_id is None else job_id,
            job_name=record.job_name if job_name is None else job_name,
            failure_reason=failure_reason,
            prediction=prediction,
        )
        changed = (
            updated.status != record.status
            or updated.job_id != record.job_id
            or updated.job_name != record.job_name
            or updated.scheduler_state != record.scheduler_state
            or updated.failure_reason != record.failure_reason
            or prediction is not None
        )
        if changed:
            self._append_record(updated, event_type="status")
        return updated

    def _fail_or_retry(
        self,
        record: ValidationExecutionRecord,
        reason: str,
        scheduler_state: SchedulerJobState,
    ) -> ValidationExecutionRecord:
        failed = self._save_status(
            record,
            "failed",
            scheduler_state=scheduler_state.value,
            failure_reason=reason,
        )
        if record.attempt_number >= self.max_attempts:
            return failed
        retry = ValidationExecutionRecord(
            validation_run_id=self.validation_run_id,
            case=record.case,
            attempt_id=self._attempt_id(record.case.case_id, record.attempt_number + 1),
            attempt_number=record.attempt_number + 1,
            status="pending",
            job_name=record.job_name,
        )
        self._append_record(retry, event_type="retry")
        return retry

    @staticmethod
    def _job_state(value: Any) -> SchedulerJobState:
        raw = getattr(value, "state", value)
        if isinstance(raw, SchedulerJobState):
            return raw
        try:
            return SchedulerJobState(str(raw).lower())
        except ValueError as exc:
            raise StateError(f"Scheduler returned an unknown validation state: {raw!r}") from exc

    def _submit(self, record: ValidationExecutionRecord) -> ValidationExecutionRecord:
        command = getattr(self.backend, "command", None)
        if command is None:
            command = getattr(self.backend, "execution_command", None)
        if command is None:
            command = ("gpumd",)
        job_name = record.job_name or (
            f"{self.job_name_prefix or 'validation'}-{record.case.ordinal:04d}-a{record.attempt_number:03d}"
        )
        try:
            submission = self.scheduler.submit(
                tuple(command),
                cwd=record.case.working_directory,
                resources=self.resources,
            )
        except Exception as exc:
            return self._fail_or_retry(
                record,
                f"submission:{type(exc).__name__}:{exc}",
                SchedulerJobState.FAILED,
            )
        job_id = getattr(submission, "job_id", None)
        if job_id is None and isinstance(submission, Mapping):
            job_id = submission.get("job_id")
        if not job_id:
            raise StateError("scheduler submission did not return a validation job ID")
        return self._save_status(
            record,
            "submitted",
            job_id=str(job_id),
            job_name=job_name,
            scheduler_state="submitted",
        )

    def _reconcile_submitted(self, record: ValidationExecutionRecord) -> ValidationExecutionRecord:
        if not record.job_id:
            raise StateError(f"Submitted validation attempt {record.attempt_id} has no scheduler job ID")
        result = self.scheduler.reconcile(record.job_id)
        job = getattr(result, "job", result)
        state = self._job_state(job)
        if state == SchedulerJobState.UNKNOWN:
            raise StateError(f"Scheduler returned an unknown state for validation job {record.job_id}")
        if state == SchedulerJobState.PENDING:
            return self._save_status(record, "submitted", scheduler_state=state.value)
        if state == SchedulerJobState.RUNNING:
            return self._save_status(record, "running", scheduler_state=state.value)
        if state == SchedulerJobState.COMPLETED:
            request = record.case.static_prediction_request(self.model)
            parser = getattr(self.backend, "parse_prediction", None)
            if not callable(parser):
                parser = getattr(self.backend, "parse_output", None)
            if not callable(parser):
                raise StateError("validation backend does not expose parse_prediction/parse_output")
            try:
                prediction = parser(request, record.case.output_path)
            except Exception as exc:
                return self._fail_or_retry(
                    record,
                    f"completed validation job produced unusable output: {exc}",
                    state,
                )
            if not isinstance(prediction, StaticPrediction):
                return self._fail_or_retry(
                    record,
                    "completed validation job returned an invalid prediction object",
                    state,
                )
            return self._save_status(
                record,
                "completed",
                scheduler_state=state.value,
                prediction=prediction,
            )
        reason = getattr(job, "reason", None) or f"scheduler state {state.value}"
        return self._fail_or_retry(record, str(reason), state)

    def _derived_status(self, records: Sequence[ValidationExecutionRecord]) -> str:
        if all(record.status == "completed" for record in records):
            return "completed"
        if any(record.status in _ACTIVE or record.status == "pending" for record in records):
            return "running"
        return "failed"

    def reconcile_once(self) -> ValidationReconciliationResult:
        """Apply one scheduler/backend state pass without guessing outcomes."""

        current = self.cases()
        active_jobs = self.scheduler.list_active_jobs(name_prefix=self.job_name_prefix).active_jobs
        submitted_this_pass = 0
        reconciled: list[ValidationExecutionRecord] = []
        for record in current:
            if record.status in _TERMINAL:
                reconciled.append(record)
                continue
            if record.status in _ACTIVE:
                reconciled.append(self._reconcile_submitted(record))
                continue
            if record.status in {"pending", "prepared"}:
                if len(active_jobs) + submitted_this_pass >= self.max_concurrent:
                    reconciled.append(record)
                    continue
                reconciled.append(self._submit(record))
                submitted_this_pass += 1
                continue
            raise StateError(f"Unknown validation execution status: {record.status}")

        status = self._derived_status(reconciled)
        self._save_run_status(status)
        return ValidationReconciliationResult(self.validation_run_id, status, tuple(reconciled))

    # A short verb is useful to workflow drivers that execute one pass at a time.
    reconcile = reconcile_once


# Public aliases keep the state-machine boundary easy to discover alongside
# the existing TrainingCampaign and DftReconciliationOrchestrator names.
ValidationCampaign = ValidationReconciliationOrchestrator
ValidationReconciler = ValidationReconciliationOrchestrator


__all__ = [
    "ValidationAttempt",
    "ValidationCampaign",
    "ValidationCaseExecution",
    "ValidationExecutionRecord",
    "ValidationReconciler",
    "ValidationReconciliationOrchestrator",
    "ValidationReconciliationResult",
]
