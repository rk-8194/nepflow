"""Authoritative identity and StateStore persistence for selection runs."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nepflow.config.models import SelectionConfig
from nepflow.domain.identities import calculate_candidate_id, calculate_structure_id
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import to_jsonable
from nepflow.state import StateStore

from .models import SelectionResult

SELECTION_RUN_SCHEMA = "selection-run-v2"
LEGACY_SELECTION_RUN_SCHEMA = "selection-run-v1"


def _structure_info(structure: Any) -> Mapping[str, Any]:
    info = getattr(structure, "info", None)
    return info if isinstance(info, Mapping) else {}


def _identity_value(structure: Any, name: str) -> str | None:
    value = getattr(structure, name, None)
    if value is None:
        value = _structure_info(structure).get(name)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise StateError(f"Selection candidate {name} must not be blank")
    return text


def structure_ids(structures: Sequence[Any]) -> list[str]:
    """Return ordered physical structure identities for a candidate sequence.

    Physical IDs are intentionally allowed to repeat: magnetic candidates can
    share one geometry while carrying different candidate identities.
    """

    result: list[str] = []
    for structure in structures:
        try:
            result.append(calculate_structure_id(structure))
        except (AttributeError, TypeError, ValueError) as exc:
            raise StateError(
                "Selection candidates must be real ASE structures with physical identities"
            ) from exc
    return result


def candidate_ids(structures: Sequence[Any]) -> list[str]:
    """Return the authoritative ordered candidate IDs for selection.

    Older non-magnetic artifacts have no explicit candidate ID.  Their
    deterministic compatibility rule is ``candidate_id == structure_id``.
    Magnetic state metadata is sufficient to derive the versioned candidate ID
    when an artifact predates explicit candidate-ID serialization.
    """

    result: list[str] = []
    physical_ids = structure_ids(structures)
    for structure, structure_id in zip(structures, physical_ids):
        explicit = _identity_value(structure, "candidate_id")
        if explicit is not None:
            result.append(explicit)
            continue
        state_id = _identity_value(structure, "magnetic_state_id")
        result.append(calculate_candidate_id(structure_id, state_id))
    if len(set(result)) != len(result):
        raise StateError("Selection candidate IDs are not unique")
    return result


def _validate_id_sequence(values: Sequence[str], *, label: str, unique: bool = False) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise StateError(f"Selection {label} IDs must be a sequence")
    result = [str(value).strip() for value in values]
    if any(not value for value in result):
        raise StateError(f"Selection {label} IDs must not be blank")
    if unique and len(set(result)) != len(result):
        raise StateError(f"Selection {label} IDs are not unique")
    return result


def selection_policy(settings: SelectionConfig) -> dict[str, Any]:
    """Serialize the complete typed selection policy used for a run."""

    return to_jsonable(asdict(settings))


def candidate_set_fingerprint(candidate_ids: Sequence[str]) -> str:
    values = _validate_id_sequence(candidate_ids, label="candidate", unique=True)
    return sha256_canonical_json(
        {"schema_version": SELECTION_RUN_SCHEMA, "candidate_ids": sorted(values)}
    )


def _legacy_candidate_set_fingerprint(candidate_ids: Sequence[str]) -> str:
    values = _validate_id_sequence(candidate_ids, label="candidate", unique=True)
    return sha256_canonical_json(
        {"schema_version": LEGACY_SELECTION_RUN_SCHEMA, "structure_ids": sorted(values)}
    )


def _selection_run_id(
    schema: str,
    project_id: str,
    candidate_ids: Sequence[str],
    settings: SelectionConfig,
) -> str:
    values = _validate_id_sequence(candidate_ids, label="candidate", unique=True)
    fingerprint = (
        candidate_set_fingerprint(values)
        if schema == SELECTION_RUN_SCHEMA
        else _legacy_candidate_set_fingerprint(values)
    )
    return "selection_" + sha256_canonical_json(
        {
            "schema_version": schema,
            "project_id": project_id,
            "candidate_set_fingerprint": fingerprint,
            "policy": selection_policy(settings),
        }
    )


def selection_run_id(
    project_id: str,
    candidate_ids: Sequence[str],
    settings: SelectionConfig,
) -> str:
    """Build an idempotent run identity from project, candidates, and policy."""

    return _selection_run_id(SELECTION_RUN_SCHEMA, project_id, candidate_ids, settings)


def legacy_selection_run_id(
    project_id: str,
    candidate_ids: Sequence[str],
    settings: SelectionConfig,
) -> str:
    """Return the v1 run ID used by pre-candidate-identity selection."""

    return _selection_run_id(LEGACY_SELECTION_RUN_SCHEMA, project_id, candidate_ids, settings)


def _persistable_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def selection_parameters(
    settings: SelectionConfig,
    candidate_ids: Sequence[str],
    result: SelectionResult,
    coverage_metrics: Mapping[str, Any] | None = None,
    *,
    candidate_structure_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the complete scientific selection record payload."""

    ordered_candidate_ids = _validate_id_sequence(candidate_ids, label="candidate", unique=True)
    ordered_structure_ids = _validate_id_sequence(
        candidate_structure_ids if candidate_structure_ids is not None else candidate_ids,
        label="structure",
    )
    if len(ordered_candidate_ids) != len(ordered_structure_ids):
        raise StateError("Selection candidate and structure ID counts do not match")

    def ids(indices: Sequence[int], values: Sequence[str]) -> list[str]:
        return [values[index] for index in indices]

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
        "candidate_set_fingerprint": candidate_set_fingerprint(ordered_candidate_ids),
        "candidate_ids": ordered_candidate_ids,
        "candidate_structure_ids": ordered_structure_ids,
        "policy": selection_policy(settings),
        "selected_candidate_ids": {
            "train": ids(result.train_indices, ordered_candidate_ids),
            "test": ids(result.test_indices, ordered_candidate_ids),
        },
        "selected_structure_ids": {
            "train": ids(result.train_indices, ordered_structure_ids),
            "test": ids(result.test_indices, ordered_structure_ids),
        },
        "mandatory_anchor_candidate_ids": {
            "seed": ids(result.seed_indices, ordered_candidate_ids),
            "single_element_elastic": ids(
                result.single_element_elastic_indices, ordered_candidate_ids
            ),
            "elastic": ids(result.elastic_indices, ordered_candidate_ids),
        },
        "mandatory_anchor_structure_ids": {
            "seed": ids(result.seed_indices, ordered_structure_ids),
            "single_element_elastic": ids(
                result.single_element_elastic_indices, ordered_structure_ids
            ),
            "elastic": ids(result.elastic_indices, ordered_structure_ids),
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
    candidate_structure_ids: Sequence[str] | None = None,
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
            candidate_structure_ids=candidate_structure_ids,
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
    values = _validate_id_sequence(ids, label=label)
    index_by_id = {identity: index for index, identity in enumerate(candidate_ids)}
    if len(index_by_id) != len(candidate_ids):
        raise StateError("Selection candidate IDs are not unique")
    missing = [identity for identity in values if identity not in index_by_id]
    if missing:
        raise StateError(
            f"Persisted selection {label} identity is absent from the current candidates: "
            f"{missing[0]}"
        )
    return [index_by_id[identity] for identity in values]


def restore_selection_result(
    record: Mapping[str, Any],
    descriptors: Any,
    candidate_ids: Sequence[str],
    *,
    candidate_structure_ids: Sequence[str] | None = None,
) -> SelectionResult:
    """Reconcile a completed record by candidate identity, never by file index."""

    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StateError("Persisted selection record has no structured parameters")
    schema = parameters.get("schema_version")
    if schema not in {SELECTION_RUN_SCHEMA, LEGACY_SELECTION_RUN_SCHEMA}:
        raise StateError(f"Unsupported persisted selection schema: {schema!r}")
    current_candidate_ids = _validate_id_sequence(candidate_ids, label="candidate", unique=True)
    current_structure_ids = (
        None
        if candidate_structure_ids is None
        else _validate_id_sequence(candidate_structure_ids, label="structure")
    )
    if current_structure_ids is not None and len(current_structure_ids) != len(
        current_candidate_ids
    ):
        raise StateError("Selection candidate and structure ID counts do not match")
    expected_fingerprint = (
        candidate_set_fingerprint(current_candidate_ids)
        if schema == SELECTION_RUN_SCHEMA
        else _legacy_candidate_set_fingerprint(current_candidate_ids)
    )
    if parameters.get("candidate_set_fingerprint") != expected_fingerprint:
        raise StateError(
            "Persisted selection candidate-set fingerprint does not match current input"
        )
    persisted_candidates = parameters.get("candidate_ids")
    if not isinstance(persisted_candidates, list):
        persisted_candidates = parameters.get("candidate_structure_ids")
    if not isinstance(persisted_candidates, list):
        raise StateError("Persisted selection candidate IDs are missing")
    persisted_candidates = _validate_id_sequence(
        persisted_candidates,
        label="candidate",
        unique=True,
    )
    if sorted(persisted_candidates) != sorted(current_candidate_ids):
        raise StateError("Persisted selection candidate identity does not match current input")
    persisted_structures = parameters.get("candidate_structure_ids")
    if persisted_structures is None:
        persisted_structures = persisted_candidates
    if not isinstance(persisted_structures, list):
        raise StateError("Persisted selection structure IDs are missing")
    persisted_structures = _validate_id_sequence(persisted_structures, label="structure")
    if len(persisted_structures) != len(persisted_candidates):
        raise StateError("Persisted selection candidate and structure ID counts do not match")
    if current_structure_ids is not None:
        persisted_by_candidate = dict(zip(persisted_candidates, persisted_structures))
        current_by_candidate = dict(zip(current_candidate_ids, current_structure_ids))
        if persisted_by_candidate != current_by_candidate:
            raise StateError("Persisted selection physical identities do not match current input")

    selected = parameters.get("selected_candidate_ids")
    if selected is None:
        selected = parameters.get("selected_structure_ids")
    selected_structures = parameters.get("selected_structure_ids")
    anchors = parameters.get("mandatory_anchor_candidate_ids")
    if anchors is None:
        anchors = parameters.get("mandatory_anchor_structure_ids")
    anchor_structures = parameters.get("mandatory_anchor_structure_ids")
    metrics = parameters.get("metrics")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(anchors, Mapping)
        or not isinstance(metrics, Mapping)
    ):
        raise StateError("Persisted selection record is incomplete")

    train_ids = selected.get("train", ())
    test_ids = selected.get("test", ())
    train_indices = _indices_for_ids(train_ids, current_candidate_ids, label="train")
    test_indices = _indices_for_ids(test_ids, current_candidate_ids, label="test")
    if set(train_indices) & set(test_indices):
        raise StateError("Persisted selection train/test identities overlap")
    seed_indices = _indices_for_ids(
        anchors.get("seed", ()), current_candidate_ids, label="seed anchor"
    )
    single_indices = _indices_for_ids(
        anchors.get("single_element_elastic", ()),
        current_candidate_ids,
        label="single-element anchor",
    )
    elastic_indices = _indices_for_ids(
        anchors.get("elastic", ()), current_candidate_ids, label="elastic anchor"
    )

    def verify_structure_selection(
        persisted: Any,
        indices: Sequence[int],
        label: str,
    ) -> None:
        if current_structure_ids is None or persisted is None:
            return
        if not isinstance(persisted, list):
            raise StateError(f"Persisted selection {label} structure IDs are malformed")
        expected = [current_structure_ids[index] for index in indices]
        actual = _validate_id_sequence(persisted, label=f"{label} structure")
        if actual != expected:
            raise StateError(f"Persisted selection {label} physical identities do not match")

    verify_structure_selection(
        selected_structures.get("train") if isinstance(selected_structures, Mapping) else None,
        train_indices,
        "train",
    )
    verify_structure_selection(
        selected_structures.get("test") if isinstance(selected_structures, Mapping) else None,
        test_indices,
        "test",
    )
    verify_structure_selection(
        anchor_structures.get("seed") if isinstance(anchor_structures, Mapping) else None,
        seed_indices,
        "seed anchor",
    )
    verify_structure_selection(
        anchor_structures.get("single_element_elastic")
        if isinstance(anchor_structures, Mapping)
        else None,
        single_indices,
        "single-element anchor",
    )
    verify_structure_selection(
        anchor_structures.get("elastic") if isinstance(anchor_structures, Mapping) else None,
        elastic_indices,
        "elastic anchor",
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
    "LEGACY_SELECTION_RUN_SCHEMA",
    "candidate_set_fingerprint",
    "candidate_ids",
    "legacy_selection_run_id",
    "persist_selection_result",
    "restore_selection_result",
    "selection_parameters",
    "selection_policy",
    "selection_run_id",
    "structure_ids",
]
