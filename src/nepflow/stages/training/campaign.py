"""StateStore-backed NEP training campaign reconciliation.

The campaign owns candidate and attempt transitions.  It never infers
successful training from scheduler disappearance and it never promotes a
model merely because a backend artifact exists.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity
from nepflow.domain.models import ModelRunRecord
from nepflow.errors import StateError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.resources import JobResources
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import sha256_file
from nepflow.io.json import to_jsonable
from nepflow.mlip.backend import (
    CollectedModelArtifacts,
    MlipBackend,
    TrainingInput,
)
from nepflow.mlip.nep.artifacts import (
    read_model_run_manifest,
    update_model_run_status,
    write_model_run_manifest,
)

from .execution import TrainingExecution
from .optimisation import CandidateConfiguration

logger = logging.getLogger(__name__)


_ACTIVE = frozenset({"submitting", "submitted", "running"})
_TERMINAL = frozenset({"completed", "failed"})


@dataclass(frozen=True, slots=True)
class TrainingAttempt:
    """Immutable persisted scheduler attempt for one model candidate."""
    attempt_id: str
    model_run_id: str
    attempt_number: int
    status: str
    job_id: str | None = None
    job_name: str | None = None
    scheduler_state: str | None = None
    progress: Mapping[str, Any] | None = None
    failure_reason: str | None = None
    execution_directory: Path | None = None
    script_path: Path | None = None
    execution_config_hash: str | None = None
    execution_config: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    """Identity-bound model candidate and its latest execution projection."""
    campaign_id: str
    dataset_id: str
    model_run_id: str
    ordinal: int
    status: str
    run_directory: Path
    hyperparameters: Mapping[str, Any]
    nep_in_sha256: str
    hyperparameters_hash: str
    script_path: Path
    attempt_number: int = 0
    job_id: str | None = None
    progress: Mapping[str, Any] | None = None
    failure_reason: str | None = None
    training_input: TrainingInput | None = field(default=None, compare=False, repr=False)
    dataset_path: Path | None = field(default=None, compare=False)

    @property
    def active(self) -> bool:
        """Return whether this candidate has an in-flight attempt."""
        return self.status in _ACTIVE

    @property
    def terminal(self) -> bool:
        """Return whether this candidate reached completed or failed state."""
        return self.status in _TERMINAL


@dataclass(frozen=True, slots=True)
class CampaignReconciliationResult:
    """Campaign status after reconciling all candidate attempts."""
    campaign_id: str
    status: str
    candidates: tuple[TrainingCandidate, ...]
    promotion_decision: Mapping[str, Any] | None = None

    @property
    def all_terminal(self) -> bool:
        """Return whether every candidate is terminal."""
        return bool(self.candidates) and all(candidate.terminal for candidate in self.candidates)

    @property
    def all_successful(self) -> bool:
        """Return whether every candidate completed with a verified artifact."""
        return bool(self.candidates) and all(
            candidate.status == "completed" for candidate in self.candidates
        )

    @property
    def ready_for_validation(self) -> bool:
        """Return whether the campaign is terminal with at least one model."""
        return self.all_terminal and any(
            candidate.status == "completed" for candidate in self.candidates
        )

    @property
    def counts(self) -> dict[str, int]:
        """Return a count of candidates by persisted status."""
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            counts[candidate.status] = counts.get(candidate.status, 0) + 1
        return counts


class TrainingCampaign:
    """Reconcile immutable model candidates through scheduler/backend evidence.

    The campaign owns append-only candidate/attempt transitions and bounded
    retries.  A completed scheduler job is not a completed model until the
    backend parses and hash-verifies the active attempt artifact; promotion is
    represented separately and is never inferred by reconciliation.
    """

    def __init__(
        self,
        *,
        campaign_id: str,
        dataset_id: str,
        state_store: Any,
        scheduler: Any,
        backend: MlipBackend,
        working_directory: str | Path,
        max_concurrent: int = 1,
        max_attempts: int = 1,
        resources: JobResources | None = None,
        dataset_path: str | Path | None = None,
    ) -> None:
        if not campaign_id.strip():
            raise ValueError("campaign_id must not be blank")
        if not dataset_id.strip():
            raise ValueError("dataset_id must not be blank")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not callable(getattr(state_store, "append_event", None)) and not callable(
            getattr(state_store, "record_training_event", None)
        ):
            raise TypeError("TrainingCampaign requires an append-only StateStore event API")
        self.campaign_id = campaign_id
        self.dataset_id = dataset_id
        self.state_store = state_store
        self.scheduler = scheduler
        self.backend = backend
        self.working_directory = Path(working_directory)
        self.max_concurrent = max_concurrent
        self.max_attempts = max_attempts
        self.resources = resources
        self.dataset_path = None if dataset_path is None else Path(dataset_path)
        self._inputs: dict[str, TrainingInput] = {}
        self._execution = TrainingExecution(
            backend=self.backend,
            resources=self.resources,
            input_for=self._input_for,
        )

    @classmethod
    def create(
        cls,
        *,
        campaign_id: str,
        dataset_id: str,
        state_store: Any,
        scheduler: Any,
        backend: MlipBackend,
        working_directory: str | Path,
        specification: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TrainingCampaign":
        """Construct a campaign and persist/validate its explicit specification."""
        campaign = cls(
            campaign_id=campaign_id,
            dataset_id=dataset_id,
            state_store=state_store,
            scheduler=scheduler,
            backend=backend,
            working_directory=working_directory,
            **kwargs,
        )
        campaign._ensure_campaign(specification)
        return campaign

    def _append(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        recorder = getattr(self.state_store, "record_training_event", None)
        event_number = len(self._events(entity_type, entity_id))
        event_id = f"training:{entity_type}:{entity_id}:{event_number:012d}"
        occurred_at = datetime.now(timezone.utc)
        existing_times: list[str] = []
        for event in self._events(entity_type, entity_id):
            value = event.get("occurred_at")
            if isinstance(value, str):
                existing_times.append(value)
        if existing_times:
            try:
                latest = max(datetime.fromisoformat(value) for value in existing_times)
                if occurred_at <= latest:
                    occurred_at = latest + timedelta(microseconds=1)
            except ValueError as exc:
                raise StateError(
                    f"Training event history for {entity_type}/{entity_id} contains "
                    "an invalid occurred_at timestamp"
                ) from exc
        occurred_text = occurred_at.isoformat()
        if callable(recorder):
            try:
                recorder(
                    entity_type,
                    entity_id,
                    event_type,
                    to_jsonable(dict(payload)),
                    event_id=event_id,
                    occurred_at=occurred_text,
                )
            except TypeError:
                # Keep lightweight event-store fakes compatible with the
                # canonical recorder while retaining the durable ordering on
                # StateStore.
                recorder(entity_type, entity_id, event_type, to_jsonable(dict(payload)))
            return
        self.state_store.append_event(
            event_id,
            entity_type,
            entity_id,
            event_type,
            to_jsonable(dict(payload)),
            occurred_at=occurred_text,
        )

    def _events(self, entity_type: str, entity_id: str | None = None) -> list[dict[str, Any]]:
        def rows(value: object) -> list[dict[str, Any]]:
            if not isinstance(value, (list, tuple)):
                raise StateError("StateStore event API returned a non-list payload")
            if not all(isinstance(item, Mapping) for item in value):
                raise StateError("StateStore event API returned a malformed row")
            return [dict(item) for item in value]

        lister = getattr(self.state_store, "list_training_events", None)
        if callable(lister):
            return rows(lister(entity_type, entity_id))
        return rows(self.state_store.list_events(entity_type=entity_type, entity_id=entity_id))

    def _ensure_campaign(self, specification: Mapping[str, Any] | None = None) -> None:
        events = self._events("training_campaign", self.campaign_id)
        if not events:
            self._create_campaign(specification)
            return
        latest = events[-1].get("payload", {})
        if latest.get("dataset_id") != self.dataset_id:
            raise StateError(
                f"Training campaign {self.campaign_id} is owned by a different dataset"
            )
        requested = {} if specification is None else to_jsonable(dict(specification))
        persisted = latest.get("specification", {})
        if not isinstance(persisted, Mapping):
            raise StateError(f"Training campaign {self.campaign_id} has a malformed specification")
        if requested:
            self._update_campaign_policy(latest, persisted, requested)

    def _create_campaign(self, specification: Mapping[str, Any] | None) -> None:
        self._append(
            "training_campaign",
            self.campaign_id,
            "created",
            {
                "campaign_id": self.campaign_id,
                "dataset_id": self.dataset_id,
                "status": "pending",
                "specification": {} if specification is None else dict(specification),
                "promotion_decision": None,
            },
        )

    def _update_campaign_policy(
        self,
        latest: Mapping[str, Any],
        persisted: Mapping[str, Any],
        requested: Mapping[str, Any],
    ) -> None:
        for key in ("candidate_keys", "candidate_matrix"):
            if key in persisted and key in requested and persisted[key] != requested[key]:
                raise StateError(
                    f"Training campaign {self.campaign_id} has a conflicting immutable specification"
                )
        if "candidate_count" in persisted and "candidate_keys" in requested:
            if int(persisted["candidate_count"]) != len(requested["candidate_keys"]):
                raise StateError(
                    f"Training campaign {self.campaign_id} has a conflicting candidate count"
                )
        if (
            "candidate_count" in persisted
            and "candidate_count" in requested
            and int(persisted["candidate_count"]) != int(requested["candidate_count"])
        ):
            raise StateError(
                f"Training campaign {self.campaign_id} has a conflicting candidate count"
            )
        merged = dict(persisted)
        merged.update(requested)
        policy_keys = {
            key
            for key in set(merged) | set(requested)
            if key
            not in {"candidate_keys", "candidate_matrix", "candidate_count", "schema_version"}
        }
        changed_policy = any(
            persisted.get(key) != requested.get(key) for key in policy_keys if key in requested
        )
        missing_policy = any(key not in persisted for key in requested if key in policy_keys)
        if not (changed_policy or missing_policy):
            return
        self._append(
            "training_campaign",
            self.campaign_id,
            "policy_updated",
            {
                **dict(latest),
                "status": latest.get("status", "pending"),
                "specification": merged,
                "policy_transition": {
                    "from": {key: persisted.get(key) for key in policy_keys if key in persisted},
                    "to": {key: requested.get(key) for key in policy_keys if key in requested},
                },
            },
        )

    def ensure(self, specification: Mapping[str, Any] | None = None) -> None:
        """Ensure the campaign identity and policy exist without new runs."""

        self._ensure_campaign(specification)

    def _campaign_payload(self) -> dict[str, Any]:
        events = self._events("training_campaign", self.campaign_id)
        if not events:
            self._ensure_campaign()
            events = self._events("training_campaign", self.campaign_id)
        return dict(events[-1].get("payload", {}))

    def _candidate_payloads(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._events("training_candidate", self.campaign_id):
            payload = event.get("payload", {})
            if payload.get("campaign_id") == self.campaign_id and payload.get("model_run_id"):
                latest[str(payload["model_run_id"])] = dict(payload)
        return latest

    def _attempt_payloads(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._events("training_attempt", self.campaign_id):
            payload = event.get("payload", {})
            if payload.get("campaign_id") == self.campaign_id and payload.get("attempt_id"):
                latest[str(payload["attempt_id"])] = dict(payload)
        return latest

    def _candidate_from_payload(self, payload: Mapping[str, Any]) -> TrainingCandidate:
        model_run_id = str(payload["model_run_id"])
        return TrainingCandidate(
            campaign_id=self.campaign_id,
            dataset_id=str(payload["dataset_id"]),
            model_run_id=model_run_id,
            ordinal=int(payload["ordinal"]),
            status=str(payload["status"]),
            run_directory=Path(str(payload["run_directory"])),
            hyperparameters=MappingProxyType(dict(payload.get("hyperparameters", {}))),
            nep_in_sha256=str(payload["nep_in_sha256"]),
            hyperparameters_hash=str(payload["hyperparameters_hash"]),
            script_path=Path(str(payload["script_path"])),
            attempt_number=int(payload.get("attempt_number", 0)),
            job_id=payload.get("job_id"),
            progress=payload.get("progress"),
            failure_reason=payload.get("failure_reason"),
            training_input=self._inputs.get(model_run_id),
            dataset_path=(
                None if payload.get("dataset_path") is None else Path(str(payload["dataset_path"]))
            ),
        )

    def candidates(self) -> tuple[TrainingCandidate, ...]:
        """Return current candidate projections in deterministic ordinal order."""
        return tuple(
            self._candidate_from_payload(payload)
            for payload in sorted(
                self._candidate_payloads().values(),
                key=lambda item: (int(item.get("ordinal", 0)), str(item["model_run_id"])),
            )
        )

    def attempts(self, model_run_id: str | None = None) -> tuple[TrainingAttempt, ...]:
        """Return persisted attempts, optionally restricted to one model run."""
        records = []
        for payload in self._attempt_payloads().values():
            if model_run_id is not None and payload.get("model_run_id") != model_run_id:
                continue
            records.append(
                TrainingAttempt(
                    attempt_id=str(payload["attempt_id"]),
                    model_run_id=str(payload["model_run_id"]),
                    attempt_number=int(payload["attempt_number"]),
                    status=str(payload["status"]),
                    job_id=payload.get("job_id"),
                    job_name=payload.get("job_name"),
                    scheduler_state=payload.get("scheduler_state"),
                    progress=payload.get("progress"),
                    failure_reason=payload.get("failure_reason"),
                    execution_directory=(
                        None
                        if payload.get("execution_directory") is None
                        else Path(str(payload["execution_directory"]))
                    ),
                    script_path=(
                        None
                        if payload.get("script_path") is None
                        else Path(str(payload["script_path"]))
                    ),
                    execution_config_hash=payload.get("execution_config_hash"),
                    execution_config=payload.get("execution_config"),
                )
            )
        return tuple(sorted(records, key=lambda item: (item.attempt_number, item.attempt_id)))

    def create_run(
        self,
        training_input: TrainingInput,
        *,
        candidate: CandidateConfiguration | None = None,
        ordinal: int | None = None,
        script_path: str | Path | None = None,
        dataset_path: str | Path | None = None,
    ) -> TrainingCandidate:
        """Create or reuse one identity-bound model candidate run.

        Reuse is allowed only when the persisted model-run identity and input
        hashes match.  The run directory and candidate event are the durable
        restart boundary; no scheduler submission occurs here.
        """

        if not isinstance(training_input, TrainingInput):
            raise TypeError("TrainingCampaign.create_run requires a TrainingInput")
        self._validate_dataset_input(training_input)
        identity = self.backend.model_run_identity(training_input)
        if identity.dataset_id != self.dataset_id:
            raise StateError("Training candidate dataset_id does not match its campaign")
        model_run_id = identity.model_run_id
        self._inputs[model_run_id] = training_input
        existing = self._candidate_payloads().get(model_run_id)
        if existing is not None:
            return self._existing_candidate(existing, identity)

        run_directory = Path(training_input.working_directory).resolve()
        self._materialize_training_input(run_directory, training_input, identity.nep_in_sha256)
        payload = self._candidate_payload(
            training_input,
            identity,
            candidate=candidate,
            ordinal=ordinal,
            script_path=script_path,
            dataset_path=dataset_path,
            run_directory=run_directory,
        )
        self._persist_candidate(identity, payload)
        self._append("training_candidate", self.campaign_id, "prepared", payload)
        return self._candidate_from_payload(payload)

    def _validate_dataset_input(self, training_input: TrainingInput) -> None:
        get_dataset = getattr(self.state_store, "get_dataset", None)
        if not callable(get_dataset):
            return
        dataset_row = get_dataset(self.dataset_id)
        if dataset_row is None:
            raise StateError(f"Training campaign dataset is not registered: {self.dataset_id}")
        if not isinstance(dataset_row, Mapping):
            raise StateError("Training campaign dataset row is malformed")
        persisted_identity = dataset_row.get("identity_json", dataset_row.get("identity", {}))
        if (
            isinstance(persisted_identity, Mapping)
            and persisted_identity.get("dataset_id") != self.dataset_id
        ):
            raise StateError(
                f"Training campaign dataset identity is inconsistent: {self.dataset_id}"
            )
        if (
            isinstance(persisted_identity, Mapping)
            and dict(persisted_identity) != training_input.dataset.identity.to_dict()
        ):
            raise StateError(
                f"Training input dataset identity does not match StateStore: {self.dataset_id}"
            )

    def _existing_candidate(self, payload: Mapping[str, Any], identity: Any) -> TrainingCandidate:
        model_run_id = str(identity.model_run_id)
        if payload.get("nep_in_sha256") != identity.nep_in_sha256:
            raise StateError(f"Candidate identity conflict: {model_run_id}")
        candidate = self._candidate_from_payload(payload)
        self._input_for(candidate)
        return candidate

    @staticmethod
    def _materialize_training_input(
        run_directory: Path,
        training_input: TrainingInput,
        expected_hash: str,
    ) -> None:
        run_directory.mkdir(parents=True, exist_ok=True)
        input_path = run_directory / "nep.in"
        if input_path.is_file():
            if sha256_file(input_path) != expected_hash:
                raise StateError(
                    f"Rendered NEP input conflicts with model-run identity: {input_path}"
                )
        else:
            atomic_write_text(input_path, training_input.content, encoding="utf-8")
            if sha256_file(input_path) != expected_hash:
                raise StateError(
                    f"TrainingInput content conflicts with model-run identity: {input_path}"
                )

    def _candidate_payload(
        self,
        training_input: TrainingInput,
        identity: Any,
        *,
        candidate: CandidateConfiguration | None,
        ordinal: int | None,
        script_path: str | Path | None,
        dataset_path: str | Path | None,
        run_directory: Path,
    ) -> dict[str, Any]:
        effective_dataset_path = (
            Path(dataset_path) if dataset_path is not None else self.dataset_path
        )
        effective_script_path = (
            Path(script_path) if script_path is not None else run_directory / "train_nep.sh"
        ).resolve()
        hyperparameters = asdict(candidate.hyperparameters) if candidate is not None else {}
        candidate_ordinal = (
            candidate.ordinal if candidate is not None else (0 if ordinal is None else int(ordinal))
        )
        return {
            "campaign_id": self.campaign_id,
            "dataset_id": self.dataset_id,
            "model_run_id": identity.model_run_id,
            "ordinal": candidate_ordinal,
            "status": "pending",
            "run_directory": str(run_directory),
            "script_path": str(effective_script_path),
            "nep_in_sha256": identity.nep_in_sha256,
            "hyperparameters_hash": identity.hyperparameters_hash,
            "hyperparameters": hyperparameters,
            "dataset_path": (
                None if effective_dataset_path is None else str(effective_dataset_path.resolve())
            ),
            "attempt_number": 0,
            "job_id": None,
            "progress": None,
            "failure_reason": None,
        }

    def _persist_candidate(
        self,
        identity: Any,
        payload: Mapping[str, Any],
    ) -> None:
        upsert_model_run = getattr(self.state_store, "upsert_model_run", None)
        existing_model = None
        get_model_run = getattr(self.state_store, "get_model_run", None)
        if callable(get_model_run):
            existing_model = get_model_run(identity.model_run_id)
        if callable(upsert_model_run) and existing_model is None:
            upsert_model_run(
                ModelRunRecord(
                    identity=identity,
                    execution_metadata={
                        "campaign_id": self.campaign_id,
                        "dataset_path": payload["dataset_path"],
                        "potential_path": payload["run_directory"],
                    },
                ),
                status="prepared",
            )

    def _save_campaign_status(self, status: str) -> None:
        current = self._campaign_payload()
        current["status"] = status
        # Promotion is intentionally never inferred by reconciliation.
        current["promotion_decision"] = current.get("promotion_decision")
        self._append("training_campaign", self.campaign_id, "status", current)

    def _save_candidate(self, candidate: TrainingCandidate, **changes: Any) -> TrainingCandidate:
        payload = {
            "campaign_id": candidate.campaign_id,
            "dataset_id": candidate.dataset_id,
            "model_run_id": candidate.model_run_id,
            "ordinal": candidate.ordinal,
            "status": candidate.status,
            "run_directory": str(candidate.run_directory),
            "script_path": str(candidate.script_path),
            "nep_in_sha256": candidate.nep_in_sha256,
            "hyperparameters_hash": candidate.hyperparameters_hash,
            "hyperparameters": dict(candidate.hyperparameters),
            "attempt_number": candidate.attempt_number,
            "job_id": candidate.job_id,
            "progress": candidate.progress,
            "failure_reason": candidate.failure_reason,
            "dataset_path": (
                None if candidate.dataset_path is None else str(candidate.dataset_path)
            ),
        }
        payload.update(changes)
        self._append("training_candidate", self.campaign_id, "status", payload)
        return self._candidate_from_payload(payload)

    def _save_attempt(self, attempt: TrainingAttempt, **changes: Any) -> TrainingAttempt:
        payload = {
            "campaign_id": self.campaign_id,
            "attempt_id": attempt.attempt_id,
            "model_run_id": attempt.model_run_id,
            "attempt_number": attempt.attempt_number,
            "status": attempt.status,
            "job_id": attempt.job_id,
            "job_name": attempt.job_name,
            "scheduler_state": attempt.scheduler_state,
            "progress": attempt.progress,
            "failure_reason": attempt.failure_reason,
            "execution_directory": (
                None if attempt.execution_directory is None else str(attempt.execution_directory)
            ),
            "script_path": None if attempt.script_path is None else str(attempt.script_path),
            "execution_config_hash": attempt.execution_config_hash,
            "execution_config": attempt.execution_config,
        }
        payload.update(changes)
        self._append("training_attempt", self.campaign_id, "status", payload)
        return TrainingAttempt(
            attempt_id=str(payload["attempt_id"]),
            model_run_id=str(payload["model_run_id"]),
            attempt_number=int(payload["attempt_number"]),
            status=str(payload["status"]),
            job_id=payload.get("job_id"),
            job_name=payload.get("job_name"),
            scheduler_state=payload.get("scheduler_state"),
            progress=payload.get("progress"),
            failure_reason=payload.get("failure_reason"),
            execution_directory=(
                None
                if payload.get("execution_directory") is None
                else Path(str(payload["execution_directory"]))
            ),
            script_path=(
                None if payload.get("script_path") is None else Path(str(payload["script_path"]))
            ),
            execution_config_hash=payload.get("execution_config_hash"),
            execution_config=payload.get("execution_config"),
        )

    def _input_for(self, candidate: TrainingCandidate) -> TrainingInput:
        input_path = candidate.run_directory / "nep.in"
        if not input_path.is_file() or sha256_file(input_path) != candidate.nep_in_sha256:
            raise StateError(
                f"Persisted NEP input is missing or changed for {candidate.model_run_id}"
            )
        if candidate.training_input is not None:
            return candidate.training_input
        identity = ArtifactIdentity.from_file("nep_input", input_path)
        dataset = TrainingDatasetManifest(
            identity=DatasetIdentity(
                candidate.dataset_id,
                {"schema_version": "nepflow.dataset.v1", "records": []},
            ),
            records=(),
        )
        return TrainingInput(
            dataset=dataset,
            working_directory=candidate.run_directory,
            content="",
            nep_in=identity,
            hyperparameters_hash=candidate.hyperparameters_hash,
        )

    def _submit(self, candidate: TrainingCandidate) -> TrainingCandidate:
        attempts = self.attempts(candidate.model_run_id)
        current = attempts[-1] if attempts else None
        repaired = self._resume_submitted_attempt(candidate, current)
        if repaired is not None:
            return repaired
        attempt = self._prepare_submission_attempt(candidate, current)
        return self._submit_prepared_attempt(candidate, attempt)

    def _resume_submitted_attempt(
        self,
        candidate: TrainingCandidate,
        attempt: TrainingAttempt | None,
    ) -> TrainingCandidate | None:
        """Repair a durable submitted attempt without creating a duplicate job."""

        if attempt is None or attempt.status not in {"submitted", "running"} or not attempt.job_id:
            return None
        if attempt.script_path is None or attempt.execution_config_hash is None:
            script_path, config_hash, execution_config = self._execution.write_script(
                candidate, attempt
            )
            attempt = self._save_attempt(
                attempt,
                script_path=str(script_path),
                execution_directory=str(script_path.parent),
                execution_config_hash=config_hash,
                execution_config=execution_config,
            )
        else:
            script_path, config_hash, _execution_config = self._execution.write_script(
                candidate, attempt, rewrite=False
            )
            if (
                str(script_path.resolve()) != str(attempt.script_path.resolve())
                or config_hash != attempt.execution_config_hash
            ):
                raise StateError(
                    f"Persisted execution configuration changed for active attempt "
                    f"{attempt.attempt_id}"
                )
        return self._save_candidate(
            candidate,
            status=attempt.status,
            attempt_number=attempt.attempt_number,
            job_id=attempt.job_id,
            progress=attempt.progress,
            failure_reason=None,
        )

    def _prepare_submission_attempt(
        self,
        candidate: TrainingCandidate,
        current: TrainingAttempt | None,
    ) -> TrainingAttempt:
        if current is not None and current.status == "submitting":
            attempt = current
        else:
            number = 1 if current is None else current.attempt_number + 1
            attempt = TrainingAttempt(
                attempt_id=f"{candidate.model_run_id}:attempt:{number}",
                model_run_id=candidate.model_run_id,
                attempt_number=number,
                status="submitting",
                job_name=(
                    f"nepflow-{self.campaign_id}-{candidate.model_run_id[-12:]}-a{number:04d}"
                ),
            )
            self._save_attempt(attempt)

        script_path, config_hash, execution_config = self._execution.write_script(
            candidate, attempt
        )
        if attempt.execution_config_hash and attempt.execution_config_hash != config_hash:
            raise StateError(
                f"Execution configuration changed for persisted training attempt "
                f"{attempt.attempt_id}; create a new attempt explicitly"
            )
        return self._save_attempt(
            attempt,
            script_path=str(script_path),
            execution_directory=str(script_path.parent),
            execution_config_hash=config_hash,
            execution_config=execution_config,
        )

    def _submit_prepared_attempt(
        self,
        candidate: TrainingCandidate,
        attempt: TrainingAttempt,
    ) -> TrainingCandidate:
        finder = getattr(self.scheduler, "find_job_by_name", None)
        existing = finder(attempt.job_name, timeout=None) if callable(finder) else None
        if existing is not None:
            job_id = self._execution.job_id(existing)
        else:
            try:
                job_id = self._execution.submit(self.scheduler, attempt, str(attempt.job_name))
            except Exception as exc:
                # Submission is an external execution boundary.  Persist the
                # failed attempt and its explicit retry/terminal decision;
                # never make a failed submit look like a pending job.
                self._save_attempt(
                    attempt,
                    status="failed",
                    failure_reason=f"submission:{type(exc).__name__}:{exc}",
                )
                next_status = "pending" if attempt.attempt_number < self.max_attempts else "failed"
                if next_status == "failed":
                    self._persist_failed_model(candidate, str(exc))
                return self._save_candidate(
                    candidate,
                    status=next_status,
                    attempt_number=attempt.attempt_number,
                    failure_reason=str(exc),
                )
        self._save_attempt(attempt, status="submitted", job_id=job_id)
        return self._save_candidate(
            candidate,
            status="submitted",
            attempt_number=attempt.attempt_number,
            job_id=job_id,
            failure_reason=None,
        )

    @staticmethod
    def _scheduler_state(value: Any) -> tuple[SchedulerJobState, Any]:
        if isinstance(value, SchedulerJobState):
            return value, None
        job = getattr(value, "job", None)
        if job is None and isinstance(value, Mapping):
            job = value.get("job")
        if job is None:
            job = value
        state = getattr(job, "state", None)
        if state is None and isinstance(job, Mapping):
            state = job.get("state")
        if not isinstance(state, SchedulerJobState):
            state = SchedulerJobState(str(state))
        return state, job

    def _persist_completed_artifact(
        self,
        candidate: TrainingCandidate,
        collected: CollectedModelArtifacts,
        attempt: TrainingAttempt,
    ) -> None:
        if collected.model_run != self._input_for(candidate).model_run_identity:
            raise StateError("Collected NEP artifact belongs to a different model run")
        manifest_path = candidate.run_directory / "model_run_manifest.json"
        if manifest_path.is_file():
            self._persist_manifest_completed_artifact(candidate, collected, attempt, manifest_path)
            return
        self._record_completed_model(candidate, collected)

    def _persist_manifest_completed_artifact(
        self,
        candidate: TrainingCandidate,
        collected: CollectedModelArtifacts,
        attempt: TrainingAttempt,
        manifest_path: Path,
    ) -> None:
        manifest = read_model_run_manifest(manifest_path)
        artifact_path = Path(
            str(collected.artifact.model.path or manifest.get("potential_artifact_path", ""))
        )
        if not artifact_path.is_file():
            raise StateError(
                f"Collected model manifest points to a missing artifact: {artifact_path}"
            )
        persisted_model = ArtifactIdentity.from_file("nep_model", artifact_path)
        if persisted_model.sha256 != collected.artifact.model.sha256:
            raise StateError("Collected NEP artifact does not match the persisted model artifact")
        if (
            collected.artifact.nep_in is not None
            and collected.artifact.nep_in.sha256 != candidate.nep_in_sha256
        ):
            raise StateError("Collected NEP input does not match the candidate identity")
        manifest["potential_artifact_path"] = str(artifact_path.resolve())
        manifest["attempt_id"] = attempt.attempt_id
        write_model_run_manifest(manifest_path, manifest)
        update_model_run_status(
            candidate.run_directory,
            "completed",
            state_store=self.state_store,
            model_run_id=candidate.model_run_id,
        )

    def _record_completed_model(
        self,
        candidate: TrainingCandidate,
        collected: CollectedModelArtifacts,
    ) -> None:
        upsert_model_run = getattr(self.state_store, "upsert_model_run", None)
        if callable(upsert_model_run):
            upsert_model_run(
                ModelRunRecord(
                    identity=collected.model_run,
                    artifact=collected.artifact,
                    execution_metadata={
                        "campaign_id": self.campaign_id,
                        "potential_path": str(candidate.run_directory),
                        "dataset_path": (
                            None if self.dataset_path is None else str(self.dataset_path.resolve())
                        ),
                    },
                ),
                status="completed",
            )

    def _persist_failed_model(self, candidate: TrainingCandidate, reason: str) -> None:
        manifest_path = candidate.run_directory / "model_run_manifest.json"
        if manifest_path.is_file():
            update_model_run_status(
                candidate.run_directory,
                "failed",
                error=reason,
                state_store=self.state_store,
                model_run_id=candidate.model_run_id,
            )
            return
        upsert_model_run = getattr(self.state_store, "upsert_model_run", None)
        if callable(upsert_model_run):
            identity = self._input_for(candidate).model_run_identity
            upsert_model_run(
                ModelRunRecord(
                    identity=identity,
                    execution_metadata={
                        "campaign_id": self.campaign_id,
                        "potential_path": str(candidate.run_directory),
                        "failure": reason,
                    },
                ),
                status="failed",
            )

    def _fail_candidate(
        self,
        candidate: TrainingCandidate,
        attempt: TrainingAttempt,
        reason: str,
        scheduler_state: SchedulerJobState,
    ) -> TrainingCandidate:
        """Persist an attempt failure and create only an explicitly allowed retry."""
        self._save_attempt(
            attempt,
            status="failed",
            scheduler_state=scheduler_state.value,
            failure_reason=reason,
        )
        next_status = "pending" if attempt.attempt_number < self.max_attempts else "failed"
        if next_status == "failed":
            self._persist_failed_model(candidate, reason)
        return self._save_candidate(
            candidate,
            status=next_status,
            attempt_number=attempt.attempt_number,
            job_id=None,
            failure_reason=reason,
        )

    def _reconcile_active(self, candidate: TrainingCandidate) -> TrainingCandidate:
        if not candidate.job_id:
            raise StateError(
                f"Active training candidate {candidate.model_run_id} has no scheduler job ID"
            )
        result = self.scheduler.reconcile(candidate.job_id)
        state, job = self._scheduler_state(result)
        attempt = self._active_attempt(candidate)
        execution_directory = attempt.execution_directory or candidate.run_directory
        progress = self.backend.parse_progress(execution_directory)
        progress_payload = self._progress_payload(progress)
        if state is SchedulerJobState.UNKNOWN:
            raise StateError(f"Scheduler returned an unknown state for {candidate.job_id}")
        if state is SchedulerJobState.PENDING:
            return self._save_active_progress(
                candidate, attempt, state, "submitted", progress_payload
            )
        if state is SchedulerJobState.RUNNING:
            return self._save_active_progress(
                candidate, attempt, state, "running", progress_payload
            )

        # A missing queue/accounting record is not completion evidence, even
        # if an old model file happens to be present.
        if state is SchedulerJobState.NOT_FOUND:
            return self._fail_candidate(
                candidate,
                attempt,
                "scheduler_disappeared_without_accounting",
                state,
            )

        return self._reconcile_terminal_attempt(
            candidate,
            attempt,
            execution_directory,
            state,
            progress_payload,
        )

    def _active_attempt(self, candidate: TrainingCandidate) -> TrainingAttempt:
        attempts = self.attempts(candidate.model_run_id)
        if not attempts:
            raise StateError(
                f"Training candidate has no persisted attempt: {candidate.model_run_id}"
            )
        return attempts[-1]

    @staticmethod
    def _progress_payload(progress: Any) -> dict[str, Any] | None:
        if progress is None:
            return None
        return {"generation": progress.generation, "loss": progress.loss}

    def _save_active_progress(
        self,
        candidate: TrainingCandidate,
        attempt: TrainingAttempt,
        scheduler_state: SchedulerJobState,
        status: str,
        progress: dict[str, Any] | None,
    ) -> TrainingCandidate:
        self._save_attempt(
            attempt,
            status=status,
            scheduler_state=scheduler_state.value,
            progress=progress,
        )
        return self._save_candidate(candidate, status=status, progress=progress)

    def _reconcile_terminal_attempt(
        self,
        candidate: TrainingCandidate,
        attempt: TrainingAttempt,
        execution_directory: Path,
        scheduler_state: SchedulerJobState,
        progress: dict[str, Any] | None,
    ) -> TrainingCandidate:
        # Completion is read only from the active attempt directory.  A model
        # file left by an earlier failed attempt cannot complete a retry.
        completion = self.backend.parse_completion(execution_directory)
        if completion.completed:
            collected = self.backend.collect_model_artifacts(
                execution_directory,
                self._input_for(candidate),
            )
            self._persist_completed_artifact(candidate, collected, attempt)
            self._save_attempt(
                attempt,
                status="completed",
                scheduler_state=scheduler_state.value,
                progress=progress,
            )
            return self._save_candidate(
                candidate,
                status="completed",
                progress=progress,
                job_id=candidate.job_id,
                failure_reason=None,
            )

        classifier = getattr(self.backend, "classify_error", None)
        reason = (
            str(
                classifier(execution_directory)
                if callable(classifier)
                else "missing_backend_completion_evidence"
            )
            or f"scheduler_{scheduler_state.value}_without_backend_completion"
        )
        return self._fail_candidate(candidate, attempt, reason, scheduler_state)

    def _derive_status(self, candidates: Sequence[TrainingCandidate]) -> str:
        if not candidates:
            return "pending"
        if any(
            candidate.status in _ACTIVE or candidate.status in {"pending", "prepared"}
            for candidate in candidates
        ):
            return "running"
        if any(candidate.status == "completed" for candidate in candidates):
            return "completed"
        return "failed"

    def reconcile(self) -> CampaignReconciliationResult:
        """Apply one idempotent state-table reconciliation pass.

        Existing submitted/running attempts are reconciled first to avoid
        duplicate jobs after restart.  Only remaining capacity is used for new
        submissions; backend completion and artifact identity are required for
        success, while promotion remains a separate explicit decision.
        """

        self._ensure_campaign()
        current = list(self.candidates())
        for candidate in current:
            if candidate.status in _ACTIVE:
                self._reconcile_active(candidate)

        current = list(self.candidates())
        active_count = sum(candidate.active for candidate in current)
        for candidate in current:
            if candidate.status not in {"pending", "prepared"}:
                continue
            if active_count >= self.max_concurrent:
                break
            updated = self._submit(candidate)
            if updated.status in _ACTIVE:
                active_count += 1

        final = self.candidates()
        status = self._derive_status(final)
        self._save_campaign_status(status)
        return CampaignReconciliationResult(
            campaign_id=self.campaign_id,
            status=status,
            candidates=final,
            promotion_decision=self._campaign_payload().get("promotion_decision"),
        )

    def snapshot(self) -> CampaignReconciliationResult:
        """Read persisted campaign state without querying or submitting jobs."""

        self._ensure_campaign()
        candidates = self.candidates()
        return CampaignReconciliationResult(
            campaign_id=self.campaign_id,
            status=self._derive_status(candidates),
            candidates=candidates,
            promotion_decision=self._campaign_payload().get("promotion_decision"),
        )

    def promotion_decision(self) -> Mapping[str, Any] | None:
        """Return the explicit promotion decision, never inferred completion."""

        value = self._campaign_payload().get("promotion_decision")
        return None if value is None else dict(value)


__all__ = [
    "CampaignReconciliationResult",
    "TrainingAttempt",
    "TrainingCandidate",
    "TrainingCampaign",
]
