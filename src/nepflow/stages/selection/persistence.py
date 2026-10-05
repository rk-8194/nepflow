"""Authoritative identity and StateStore persistence for selection runs."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nepflow.config.models import SelectionConfig
from nepflow.domain.identities import calculate_structure_id
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import to_jsonable
from nepflow.state import StateStore

from .models import SelectionResult

SELECTION_RUN_SCHEMA = "selection-run-v1"


def structure_ids(structures: Sequence[Any]) -> list[str]:
    """Return ordered physical structure identities for a candidate sequence."""

    result: list[str] = []
    for structure in structures:
        try:
            result.append(calculate_structure_id(structure))
        except (AttributeError, TypeError, ValueError) as exc:
            raise StateError(
                "Selection candidates must be real ASE structures with physical identities"
            ) from exc
    if len(set(result)) != len(result):
        raise StateError("Selection candidate structure IDs are not unique")
    return result


def selection_policy(settings: SelectionConfig) -> dict[str, Any]:
    """Serialize the complete typed selection policy used for a run."""

    return to_jsonable(asdict(settings))


def candidate_set_fingerprint(candidate_ids: Sequence[str]) -> str:
    return sha256_canonical_json(
        {
            "schema_version": SELECTION_RUN_SCHEMA,
            "structure_ids": sorted(candidate_ids),
        }
    )


def selection_run_id(
    project_id: str,
    candidate_ids: Sequence[str],
    settings: SelectionConfig,
) -> str:
    """Build an idempotent run identity from project, candidates, and policy."""

    return "selection_" + sha256_canonical_json(
        {
            "schema_version": SELECTION_RUN_SCHEMA,
            "project_id": project_id,
            "candidate_set_fingerprint": candidate_set_fingerprint(candidate_ids),
            "policy": selection_policy(settings),
        }
    )


def _persistable_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def selection_parameters(
    settings: SelectionConfig,
    candidate_ids: Sequence[str],
    result: SelectionResult,
    coverage_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete scientific selection record payload."""

    def ids(indices: Sequence[int]) -> list[str]:
        return [candidate_ids[index] for index in indices]

    metrics = {
        "train_min_dist": _persistable_number(result.train_min_dist),
        "test_min_dist": _persistable_number(result.test_min_dist),
        "min_train_test_dist": _persistable_number(result.min_train_test_dist),
        "mean_train_test_dist": _persistable_number(result.mean_train_test_dist),
        "train_count": len(result.train_indices),
        "test_count": len(result.test_indices),
        "train_anchor_count": result.train_anchor_count,
        "train_fps_count": result.train_fps_count,
        "train_seed_count": result.train_seed_count,
        "train_single_element_elastic_count": result.train_single_element_elastic_count,
        "train_elastic_count": result.train_elastic_count,
    }
    return {
        "schema_version": SELECTION_RUN_SCHEMA,
        "candidate_set_fingerprint": candidate_set_fingerprint(candidate_ids),
        "candidate_structure_ids": list(candidate_ids),
        "policy": selection_policy(settings),
        "selected_structure_ids": {
            "train": ids(result.train_indices),
            "test": ids(result.test_indices),
        },
        "mandatory_anchor_structure_ids": {
            "seed": ids(result.seed_indices),
            "single_element_elastic": ids(result.single_element_elastic_indices),
            "elastic": ids(result.elastic_indices),
        },
        "metrics": metrics,
        "coverage_metrics": to_jsonable(coverage_metrics or {}),
    }


def persist_selection_result(
    store: StateStore,
    project_id: str,
    project_name: str,
    project_dir: str,
    settings: SelectionConfig,
    candidate_ids: Sequence[str],
    result: SelectionResult,
    *,
    coverage_metrics: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Write one completed selection record through the canonical StateStore API."""

    now = datetime.now(timezone.utc).isoformat()
    if store.get_project(project_id) is None:
        raise StateError(
            f"Cannot persist selection for unknown project {project_id!r}; "
            "initialize the authoritative project record first"
        )
    return store.upsert_selection_run(
        selection_run_id(project_id, candidate_ids, settings),
        project_id,
        status="completed",
        method="composition_aware_fps" if settings.composition_aware_fps else "fps",
        parameters=selection_parameters(
            settings,
            candidate_ids,
            result,
            coverage_metrics,
        ),
        started_at=started_at or now,
        completed_at=completed_at or now,
    )


def _indices_for_ids(
    ids: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    label: str,
) -> list[int]:
    index_by_id = {identity: index for index, identity in enumerate(candidate_ids)}
    if len(index_by_id) != len(candidate_ids):
        raise StateError("Selection candidate structure IDs are not unique")
    missing = [identity for identity in ids if identity not in index_by_id]
    if missing:
        raise StateError(
            f"Persisted selection {label} identity is absent from the current candidates: "
            f"{missing[0]}"
        )
    return [index_by_id[identity] for identity in ids]


def restore_selection_result(
    record: Mapping[str, Any],
    descriptors: Any,
    candidate_ids: Sequence[str],
) -> SelectionResult:
    """Reconcile a completed record by structure identity, never by file index."""

    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StateError("Persisted selection record has no structured parameters")
    if parameters.get("schema_version") != SELECTION_RUN_SCHEMA:
        raise StateError(
            f"Unsupported persisted selection schema: {parameters.get('schema_version')!r}"
        )
    if parameters.get("candidate_set_fingerprint") != candidate_set_fingerprint(candidate_ids):
        raise StateError(
            "Persisted selection candidate-set fingerprint does not match current input"
        )
    persisted_candidates = parameters.get("candidate_structure_ids")
    if sorted(persisted_candidates or ()) != sorted(candidate_ids):
        raise StateError("Persisted selection candidate identity does not match current input")
    selected = parameters.get("selected_structure_ids")
    anchors = parameters.get("mandatory_anchor_structure_ids")
    metrics = parameters.get("metrics")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(anchors, Mapping)
        or not isinstance(metrics, Mapping)
    ):
        raise StateError("Persisted selection record is incomplete")

    train_indices = _indices_for_ids(selected.get("train", ()), candidate_ids, label="train")
    test_indices = _indices_for_ids(selected.get("test", ()), candidate_ids, label="test")
    if set(train_indices) & set(test_indices):
        raise StateError("Persisted selection train/test identities overlap")
    seed_indices = _indices_for_ids(anchors.get("seed", ()), candidate_ids, label="seed anchor")
    single_indices = _indices_for_ids(
        anchors.get("single_element_elastic", ()), candidate_ids, label="single-element anchor"
    )
    elastic_indices = _indices_for_ids(
        anchors.get("elastic", ()), candidate_ids, label="elastic anchor"
    )

    def metric(name: str, default: float) -> float:
        value = metrics.get(name)
        return default if value is None else float(value)

    return SelectionResult(
        descriptors=descriptors,
        train_indices=train_indices,
        train_min_dist=metric("train_min_dist", 0.0),
        train_seed_count=int(metrics.get("train_seed_count", len(seed_indices))),
        train_single_element_elastic_count=int(
            metrics.get("train_single_element_elastic_count", len(single_indices))
        ),
        train_elastic_count=int(metrics.get("train_elastic_count", len(elastic_indices))),
        train_anchor_count=int(
            metrics.get(
                "train_anchor_count", len(set(seed_indices + single_indices + elastic_indices))
            )
        ),
        train_fps_count=int(metrics.get("train_fps_count", 0)),
        test_indices=test_indices,
        test_min_dist=metric("test_min_dist", 0.0),
        min_train_test_dist=metric("min_train_test_dist", math.inf),
        mean_train_test_dist=metric("mean_train_test_dist", math.inf),
        seed_indices=seed_indices,
        single_element_elastic_indices=single_indices,
        elastic_indices=elastic_indices,
    )


__all__ = [
    "SELECTION_RUN_SCHEMA",
    "candidate_set_fingerprint",
    "persist_selection_result",
    "restore_selection_result",
    "selection_parameters",
    "selection_policy",
    "selection_run_id",
    "structure_ids",
]
