"""Exact entropy-pool validation, frozen bandwidths, and bounded calibration."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)
from nepflow.stages.selection.representations import LocalEnvironmentRepresentation

from .checkpointing import CheckpointValidationError, VersionedCheckpointStore
from .gpu_neighbours import (
    ExactTiledGPUNeighbourIndex,
    compute_gpu_tiled_neighbours_many,
)
from .gpu_support import ExactTiledGPUSupport
from .kernels import (
    evaluate_leave_one_out_objective,
    evaluate_leave_one_out_objectives,
)
from .models import (
    BANDWIDTH_SCHEMA_VERSION,
    CALIBRATION_OPTIMIZER_ID,
    CALIBRATION_OPTIMIZER_VERSION,
    DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES,
    DEFAULT_NEIGHBOUR_BACKEND_ID,
    ENTROPY_OBJECTIVE_SCHEMA_VERSION,
    GPU_NEIGHBOUR_BACKEND_ALIASES,
    KERNEL_FAMILY,
    KERNEL_OPERATOR_SCHEMA_VERSION,
    KERNEL_VERSION,
    NEIGHBOUR_DISTINCT_POLICY,
    NEIGHBOUR_METRIC,
    BandwidthCalibrationResult,
    CalibrationAttempt,
    EntropyBandwidthSettings,
    EntropyPool,
    FrozenBandwidths,
)
from .neighbours import (
    ExactNeighbourResult,
    IndexedCPUNeighbourIndex,
    RadiusQueryCapacityError,
    TiledCPUNeighbourIndex,
    _estimate_radius_workspace_bytes,
    build_neighbour_index,
    compute_indexed_cpu_neighbours_many,
    compute_neighbours,
    compute_tiled_cpu_neighbours_many,
)

logger = logging.getLogger(__name__)


class BandwidthCalibrationError(ValueError):
    """Raised when no scientifically valid bandwidth calibration exists."""

    def __init__(self, message: str, *, attempts: Sequence[CalibrationAttempt] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


BandwidthCalibrationCapacityError = RadiusQueryCapacityError

_C_BATCH_PILOT_VERSION = "support-cost-pilot-v2"
_C_BATCH_SUPPORT_RATIO_LIMIT = 2.0
_C_BATCH_PILOT_SOURCE_LIMIT = 16
_C_BATCH_NORMALIZATION_COST = 1.0e-9
_C_BATCH_CHECKPOINT_COST_SECONDS = 1.0e-3
_C_BATCH_MEMORY_BANDWIDTH_BYTES_PER_SECOND = 20.0e9


@dataclass(frozen=True, slots=True)
class _ExplicitRow:
    candidate_id: str
    structure_id: str
    atom_index: int


def _array_fingerprint(values: np.ndarray) -> str:
    payload = {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
    }
    return sha256_canonical_json(payload)


def _row_owner(row: Any) -> str:
    if isinstance(row, str):
        owner = row.strip()
    elif isinstance(row, Mapping):
        owner = str(row.get("candidate_id", "")).strip()
    else:
        owner = str(getattr(row, "candidate_id", "")).strip()
    if not owner:
        raise ValueError("every entropy descriptor row must have exactly one candidate owner")
    return owner


def _row_identity(row: Any, index: int) -> dict[str, Any]:
    if isinstance(row, Mapping):
        structure_id = row.get("structure_id", "")
        atom_index = row.get("atom_index", index)
    else:
        structure_id = getattr(row, "structure_id", "")
        atom_index = getattr(row, "atom_index", index)
    return {
        "candidate_id": _row_owner(row),
        "structure_id": str(structure_id),
        "atom_index": int(atom_index),
        "row_index": index,
    }


def _coerce_representation_inputs(
    representation_or_descriptors: LocalEnvironmentRepresentation | EntropyPool | np.ndarray,
    rows: Sequence[Any] | None,
    candidate_ids: Sequence[str] | None,
    *,
    row_candidate_ids: Sequence[str] | None,
    transform_fingerprint: str | None,
    representation_fingerprint: str | None,
) -> tuple[np.ndarray, tuple[Any, ...], tuple[str, ...], str, str]:
    if isinstance(representation_or_descriptors, EntropyPool):
        if any(
            value is not None
            for value in (
                rows,
                candidate_ids,
                row_candidate_ids,
                transform_fingerprint,
                representation_fingerprint,
            )
        ):
            raise ValueError("explicit entropy pool cannot be combined with replacement metadata")
        return (
            representation_or_descriptors.descriptors,
            representation_or_descriptors.rows,
            representation_or_descriptors.candidate_ids,
            representation_or_descriptors.transform_fingerprint,
            representation_or_descriptors.representation_fingerprint,
        )
    if isinstance(representation_or_descriptors, LocalEnvironmentRepresentation):
        if any(value is not None for value in (rows, candidate_ids, row_candidate_ids)):
            raise ValueError(
                "local representation cannot be combined with replacement row metadata"
            )
        try:
            transform_value = representation_or_descriptors.transform_fingerprint
            representation_value = representation_or_descriptors.fingerprint
            if not isinstance(transform_value, str) or not isinstance(representation_value, str):
                raise ValueError("identity values must be strings")
            transform_id = transform_value
            representation_id = representation_value
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("local representation has an invalid transform identity") from exc
        return (
            representation_or_descriptors.descriptors,
            tuple(representation_or_descriptors.rows),
            tuple(representation_or_descriptors.candidate_ids),
            transform_id,
            representation_id,
        )

    descriptors = np.asarray(representation_or_descriptors)
    if rows is None and row_candidate_ids is None:
        raise ValueError("explicit descriptors require rows or row_candidate_ids")
    if candidate_ids is None:
        raise ValueError("explicit descriptors require candidate_ids")
    if rows is None:
        owners = tuple(str(value) for value in row_candidate_ids or ())
        rows = tuple(_ExplicitRow(owner, owner, index) for index, owner in enumerate(owners))
    else:
        rows = tuple(rows)
        if row_candidate_ids is not None:
            supplied_owners = tuple(str(value) for value in row_candidate_ids)
            if len(supplied_owners) != len(rows) or any(
                _row_owner(row) != owner for row, owner in zip(rows, supplied_owners)
            ):
                raise ValueError("row_candidate_ids conflicts with row metadata")
    if transform_fingerprint is None:
        descriptor_id = _array_fingerprint(descriptors)
    elif isinstance(transform_fingerprint, str):
        descriptor_id = transform_fingerprint
    else:
        raise ValueError("transform_fingerprint must be a string")
    if representation_fingerprint is None:
        representation_id = descriptor_id
    elif isinstance(representation_fingerprint, str):
        representation_id = representation_fingerprint
    else:
        raise ValueError("representation_fingerprint must be a string")
    return (
        descriptors,
        rows,
        tuple(str(value) for value in candidate_ids),
        descriptor_id,
        representation_id,
    )


def build_entropy_pool(
    representation_or_descriptors: LocalEnvironmentRepresentation | EntropyPool | np.ndarray,
    rows: Sequence[Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
    *,
    row_candidate_ids: Sequence[str] | None = None,
    transform_fingerprint: str | None = None,
    representation_fingerprint: str | None = None,
) -> EntropyPool:
    """Validate a local representation and derive equal-candidate row masses."""

    descriptors, ordered_rows, ordered_candidates, transform_id, representation_id = (
        _coerce_representation_inputs(
            representation_or_descriptors,
            rows,
            candidate_ids,
            row_candidate_ids=row_candidate_ids,
            transform_fingerprint=transform_fingerprint,
            representation_fingerprint=representation_fingerprint,
        )
    )
    values = np.asarray(descriptors)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("transformed descriptors must have dtype float64")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("transformed descriptors must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("transformed descriptors must be finite")
    if not ordered_candidates or any(not candidate.strip() for candidate in ordered_candidates):
        raise ValueError("candidate_ids must be non-empty and non-blank")
    if len(set(ordered_candidates)) != len(ordered_candidates):
        raise ValueError("candidate_ids must be unique")
    if len(ordered_rows) != values.shape[0]:
        raise ValueError("rows must align with transformed descriptor rows")
    if not transform_id or not representation_id:
        raise ValueError("transform and representation fingerprints must not be blank")

    candidate_indices: list[int] = []
    for row in ordered_rows:
        owner = _row_owner(row)
        try:
            candidate_indices.append(ordered_candidates.index(owner))
        except ValueError as exc:
            raise ValueError(f"unknown candidate owner {owner!r} in entropy row mapping") from exc
    counts = np.bincount(candidate_indices, minlength=len(ordered_candidates))
    if np.any(counts == 0):
        missing = ordered_candidates[int(np.flatnonzero(counts == 0)[0])]
        raise ValueError(f"candidate {missing!r} owns no local descriptor rows")
    masses = np.asarray(
        [1.0 / (len(ordered_candidates) * int(counts[index])) for index in candidate_indices],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("derived candidate probabilities must be finite and positive")
    if not math.isclose(float(np.sum(masses, dtype=np.float64)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("derived candidate probabilities must sum to one")

    row_payload = [_row_identity(row, index) for index, row in enumerate(ordered_rows)]
    pool_payload = {
        "schema": "entropy-pool-v1",
        "descriptors": _array_fingerprint(np.ascontiguousarray(values)),
        "rows": row_payload,
        "candidate_ids": list(ordered_candidates),
        "row_candidate_indices": candidate_indices,
        "probabilities": masses.tolist(),
        "transform_fingerprint": transform_id,
        "representation_fingerprint": representation_id,
    }
    pool_fingerprint = sha256_canonical_json(pool_payload)
    return EntropyPool(
        descriptors=values,
        rows=tuple(ordered_rows),
        candidate_ids=ordered_candidates,
        row_candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        probabilities=masses,
        transform_fingerprint=transform_id,
        representation_fingerprint=representation_id,
        fingerprint=pool_fingerprint,
    )


prepare_entropy_pool = build_entropy_pool


def derive_candidate_probabilities(
    descriptors: np.ndarray,
    row_candidate_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> np.ndarray:
    """Return canonical equal-candidate/equal-within-candidate row masses."""

    pool = build_entropy_pool(
        descriptors,
        candidate_ids=candidate_ids,
        row_candidate_ids=row_candidate_ids,
    )
    return pool.probabilities


candidate_weights = derive_candidate_probabilities


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and strictly positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _progress_timing(completed: int, total: int, started: float) -> tuple[float, float]:
    elapsed = max(0.0, time.perf_counter() - started)
    rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else 0.0
    return elapsed, max(0.0, remaining)


def _pool_row_ids(pool: EntropyPool) -> tuple[str, ...]:
    def row_value(row: Any, name: str, default: Any) -> Any:
        return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)

    return tuple(
        f"{_row_owner(row)}:{str(row_value(row, 'structure_id', ''))}:"
        f"{int(row_value(row, 'atom_index', index))}:{index}"
        for index, row in enumerate(pool.rows)
    )


def _freeze_from_neighbours(
    pool: EntropyPool,
    neighbours: ExactNeighbourResult,
    c: float,
) -> FrozenBandwidths:
    scale = _positive_float(c, "c")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        try:
            bandwidths = np.asarray(scale * neighbours.radii, dtype=np.float64)
        except FloatingPointError as exc:
            raise ValueError("c multiplied by neighbour radii overflowed or underflowed") from exc
    if not np.all(np.isfinite(bandwidths)) or np.any(bandwidths <= 0.0):
        raise ValueError("frozen bandwidths must be finite and strictly positive")
    identity = {
        "schema": BANDWIDTH_SCHEMA_VERSION,
        "pool": pool.fingerprint,
        "transform": pool.transform_fingerprint,
        "neighbour": neighbours.fingerprint,
        "backend": neighbours.backend,
        "backend_version": neighbours.backend_version,
        "backend_fingerprint": neighbours.backend_fingerprint,
        "metric": neighbours.metric,
        "k": neighbours.k,
        "c": scale,
    }
    return FrozenBandwidths(
        radii=neighbours.radii,
        bandwidths=bandwidths,
        k=neighbours.k,
        c=scale,
        neighbour_fingerprint=neighbours.fingerprint,
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint=sha256_canonical_json(identity),
        backend=neighbours.backend,
        metric=neighbours.metric,
        backend_version=neighbours.backend_version,
        backend_fingerprint=neighbours.backend_fingerprint,
    )


def calculate_frozen_bandwidths(
    representation_or_pool: LocalEnvironmentRepresentation | EntropyPool | np.ndarray,
    k: int,
    c: float,
    *,
    rows: Sequence[Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
    row_candidate_ids: Sequence[str] | None = None,
    transform_fingerprint: str | None = None,
    representation_fingerprint: str | None = None,
    chunk_size: int = 1024,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    max_neighbour_entries: int | None = None,
    max_index_bytes: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = None,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_neighbour_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> FrozenBandwidths:
    """Calculate and freeze exact full-pool ``r_k`` and ``h=c*r_k``."""

    pool = (
        representation_or_pool
        if isinstance(representation_or_pool, EntropyPool)
        else build_entropy_pool(
            representation_or_pool,
            rows=rows,
            candidate_ids=candidate_ids,
            row_candidate_ids=row_candidate_ids,
            transform_fingerprint=transform_fingerprint,
            representation_fingerprint=representation_fingerprint,
        )
    )
    service = resource_budget or build_resource_budget()
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("entropy bandwidth calculation"),
            operation="entropy bandwidth calculation",
        )
    if remaining < 1:
        raise ResourceCapacityError(
            "bandwidth calculation has no allocatable runtime memory headroom",
            operation="entropy bandwidth calculation",
            requested_bytes=1,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    neighbour_kwargs: dict[str, Any] = {}
    if backend in GPU_NEIGHBOUR_BACKEND_ALIASES:
        neighbour_kwargs.update(
            gpu_device=gpu_device,
            gpu_devices=gpu_devices,
            gpu_tile_size=gpu_tile_size,
            max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
        )
    index = build_neighbour_index(
        pool.descriptors,
        backend=backend,
        tile_size=chunk_size,
        max_index_bytes=(remaining if max_index_bytes is None else min(max_index_bytes, remaining)),
        max_radius_query_bytes=(
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(max_radius_query_bytes, max(1, remaining))
        ),
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        **neighbour_kwargs,
        resource_budget=service,
    )
    neighbours = compute_neighbours(
        pool.descriptors,
        k,
        backend=backend,
        index=index,
        max_neighbour_entries=max_neighbour_entries,
        chunk_size=chunk_size,
        representation_fingerprint=pool.representation_fingerprint,
        row_ids=_pool_row_ids(pool),
        **neighbour_kwargs,
        resource_budget=service,
    )
    return _freeze_from_neighbours(pool, neighbours, c)


freeze_bandwidths = calculate_frozen_bandwidths
compute_bandwidths = calculate_frozen_bandwidths
compute_frozen_bandwidths = calculate_frozen_bandwidths
calculate_bandwidths = calculate_frozen_bandwidths


def _settings_from_arguments(
    settings: EntropyBandwidthSettings | Mapping[str, Any] | None,
    *,
    mode: str | None,
    k: int | None,
    c: float | None,
    k_candidates: Sequence[int] | None,
    c_candidates: Sequence[float] | None,
    backend: str,
    metric: str,
    chunk_size: int,
    max_neighbour_entries: int | None,
    max_index_bytes: int | None,
    max_radius_query_bytes: int | None,
    max_neighbour_cache_bytes: int | None,
    max_calibration_work_bytes: int | None,
    calibration_batch_size: int | None,
    kernel_support_execution: str,
    gpu_device: int,
    gpu_devices: Sequence[int] | str | int | None,
    gpu_tile_size: int,
    max_gpu_support_bytes: int | None,
    max_gpu_neighbour_bytes: int | None,
    loo_checkpoint_every_sources: int | None,
    loo_checkpoint_every_seconds: float | None,
) -> EntropyBandwidthSettings:
    if settings is not None:
        if any(
            value is not None
            for value in (
                mode,
                k,
                c,
                k_candidates,
                c_candidates,
                loo_checkpoint_every_sources,
                loo_checkpoint_every_seconds,
            )
        ):
            raise ValueError("settings cannot be combined with explicit calibration arguments")
        if isinstance(settings, EntropyBandwidthSettings):
            return settings
        return EntropyBandwidthSettings(**dict(settings))
    selected_mode = "automatic" if mode is None else mode
    if selected_mode == "manual":
        return EntropyBandwidthSettings(
            mode=selected_mode,
            k=k,
            c=c,
            backend=backend,
            metric=metric,
            chunk_size=chunk_size,
            max_neighbour_entries=max_neighbour_entries,
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            max_calibration_work_bytes=max_calibration_work_bytes,
            calibration_batch_size=calibration_batch_size,
            kernel_support_execution=kernel_support_execution,
            gpu_device=gpu_device,
            gpu_devices=gpu_devices,
            gpu_tile_size=gpu_tile_size,
            max_gpu_support_bytes=max_gpu_support_bytes,
            max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
            loo_checkpoint_every_sources=(
                DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES
                if loo_checkpoint_every_sources is None
                else loo_checkpoint_every_sources
            ),
            loo_checkpoint_every_seconds=loo_checkpoint_every_seconds,
        )
    return EntropyBandwidthSettings(
        mode=selected_mode,
        k_candidates=tuple(k_candidates) if k_candidates is not None else (1, 2, 4, 8),
        c_candidates=tuple(c_candidates) if c_candidates is not None else (1.5, 2.0, 4.0, 8.0),
        backend=backend,
        metric=metric,
        chunk_size=chunk_size,
        max_neighbour_entries=max_neighbour_entries,
        max_index_bytes=max_index_bytes,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        max_calibration_work_bytes=max_calibration_work_bytes,
        calibration_batch_size=calibration_batch_size,
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
        max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
        loo_checkpoint_every_sources=(
            DEFAULT_LOO_CHECKPOINT_EVERY_SOURCES
            if loo_checkpoint_every_sources is None
            else loo_checkpoint_every_sources
        ),
        loo_checkpoint_every_seconds=loo_checkpoint_every_seconds,
    )


def _attempt_payload(attempt: CalibrationAttempt) -> dict[str, Any]:
    return {
        "k": int(attempt.k),
        "c": float(attempt.c),
        "status": attempt.status,
        "objective": None if attempt.objective is None else float(attempt.objective),
        "reason": attempt.reason,
        "bandwidth_fingerprint": attempt.bandwidth_fingerprint,
    }


_CALIBRATION_GRID_CHECKPOINT_NAME = "calibration-grid"
_CALIBRATION_GRID_SCHEMA = "entropy-calibration-grid-v1"


def _loo_checkpoint_name(mode: str, candidate_k: int, candidate_cs: Sequence[float]) -> str:
    ordered_cs = tuple(float(value) for value in candidate_cs)
    if mode == "manual":
        return f"loo-k{int(candidate_k)}"
    return f"loo-k{int(candidate_k)}-c-" + "-".join(f"{value:g}" for value in ordered_cs)


def _calibration_grid_identity(
    pool: EntropyPool,
    settings: EntropyBandwidthSettings,
    domain_k: Sequence[int],
    domain_c: Sequence[float],
    index: Any,
) -> dict[str, Any]:
    """Return the scientific identity for the completed calibration grid."""

    backend = str(getattr(index, "backend", settings.backend))
    backend_version = str(getattr(index, "backend_version", ""))
    backend_fingerprint = str(getattr(index, "fingerprint", ""))
    return {
        "phase": "calibration-grid",
        "schema": _CALIBRATION_GRID_SCHEMA,
        "pool_fingerprint": pool.fingerprint,
        "descriptor_fingerprint": _array_fingerprint(pool.descriptors),
        "ordered_row_fingerprint": sha256_canonical_json(list(_pool_row_ids(pool))),
        "candidate_ids_fingerprint": sha256_canonical_json(list(pool.candidate_ids)),
        "row_candidate_indices_fingerprint": _array_fingerprint(pool.row_candidate_indices),
        "probability_fingerprint": _array_fingerprint(pool.probabilities),
        "transform_fingerprint": pool.transform_fingerprint,
        "representation_fingerprint": pool.representation_fingerprint,
        "source_order": "ascending-original-pool-row-index",
        "target_order": "ascending-original-pool-row-index",
        "distinct_location_policy": str(
            getattr(index, "distinct_location_policy", NEIGHBOUR_DISTINCT_POLICY)
        ),
        "backend": backend,
        "backend_version": backend_version,
        "backend_numerical_fingerprint": backend_fingerprint,
        "metric": str(getattr(index, "metric", settings.metric)),
        "kernel_family": KERNEL_FAMILY,
        "kernel_version": KERNEL_VERSION,
        "kernel_operator_schema": KERNEL_OPERATOR_SCHEMA_VERSION,
        "loo_schema": "loo-checkpoint-v1",
        "normalization": "source-column-all-targets-including-self",
        "loo_exclusion": "source-row-only",
        "objective_schema": ENTROPY_OBJECTIVE_SCHEMA_VERSION,
        "objective_semantics": "-sum(p_a*log(loo_a)), float64 source-normalized LOO",
        "validity_semantics": "finite-positive-loo-and-finite-objective",
        "optimizer": "manual" if settings.mode == "manual" else CALIBRATION_OPTIMIZER_ID,
        "optimizer_version": (
            "manual-v1" if settings.mode == "manual" else CALIBRATION_OPTIMIZER_VERSION
        ),
        "mode": settings.mode,
        "ordered_k_domain": [int(value) for value in domain_k],
        "ordered_c_domain": [float(value) for value in domain_c],
        "tie_policy": "first-canonical-minimum-strict-less-than",
    }


def _restore_calibration_grid_state(
    state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    pool: EntropyPool,
    mode: str,
    domain_k: Sequence[int],
    domain_c: Sequence[float],
) -> dict[str, Any]:
    """Validate and decode a grid commit without trusting restored objects."""

    if state.get("schema") != _CALIBRATION_GRID_SCHEMA:
        raise CheckpointValidationError("calibration grid state schema is incompatible")
    total = len(domain_k) * len(domain_c)
    raw_attempts = state.get("attempts")
    if not isinstance(raw_attempts, list) or len(raw_attempts) > total:
        raise CheckpointValidationError("calibration grid attempts are incomplete or invalid")
    if state.get("cursor") != len(raw_attempts):
        raise CheckpointValidationError("calibration grid attempt cursor is invalid")
    attempts: list[CalibrationAttempt] = []
    for position, raw in enumerate(raw_attempts):
        if not isinstance(raw, dict):
            raise CheckpointValidationError("calibration grid attempt record is invalid")
        expected_k = int(domain_k[position // len(domain_c)])
        expected_c = float(domain_c[position % len(domain_c)])
        try:
            actual_k = int(raw["k"])
            actual_c = float(raw["c"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CheckpointValidationError(
                "calibration grid attempt coordinate is invalid"
            ) from exc
        if actual_k != expected_k or actual_c != expected_c:
            raise CheckpointValidationError(
                "calibration grid attempt order does not match the canonical search domain"
            )
        status = raw.get("status")
        if status not in {"valid", "invalid"}:
            raise CheckpointValidationError("calibration grid attempt status is invalid")
        fingerprint = raw.get("bandwidth_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or not fingerprint.strip()
        ):
            raise CheckpointValidationError("calibration grid bandwidth fingerprint is invalid")
        if status == "valid":
            objective = raw.get("objective")
            if isinstance(objective, bool) or not isinstance(objective, (int, float)):
                raise CheckpointValidationError("valid calibration objective is invalid")
            objective = float(objective)
            if not math.isfinite(objective) or raw.get("reason") is not None or not fingerprint:
                raise CheckpointValidationError("valid calibration attempt proof is incomplete")
            reason = None
        else:
            if raw.get("objective") is not None:
                raise CheckpointValidationError("invalid calibration attempt has an objective")
            reason_value = raw.get("reason")
            if not isinstance(reason_value, str) or not reason_value.strip():
                raise CheckpointValidationError("invalid calibration attempt has no reason")
            objective = None
            reason = reason_value
        attempts.append(
            CalibrationAttempt(
                actual_k,
                actual_c,
                status,
                objective=objective,
                reason=reason,
                bandwidth_fingerprint=fingerprint,
            )
        )

    raw_times = state.get("pair_wall_seconds")
    if not isinstance(raw_times, list) or len(raw_times) != len(attempts):
        raise CheckpointValidationError("calibration grid pair timing state is invalid")
    pair_wall_seconds: list[float] = []
    for value in raw_times:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CheckpointValidationError("calibration grid pair timing is invalid")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise CheckpointValidationError("calibration grid pair timing is invalid")
        pair_wall_seconds.append(seconds)

    active_raw = state.get("active_batch")
    active_batch: dict[str, Any] | None = None
    if active_raw is not None:
        if not isinstance(active_raw, dict) or len(attempts) >= total:
            raise CheckpointValidationError("calibration grid active batch is invalid")
        next_position = len(attempts)
        expected_k = int(domain_k[next_position // len(domain_c)])
        expected_c = float(domain_c[next_position % len(domain_c)])
        values = active_raw.get("c_values")
        try:
            active_k = int(active_raw["k"])
            active_cs = tuple(float(value) for value in values)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CheckpointValidationError(
                "calibration grid active batch coordinates are invalid"
            ) from exc
        if (
            active_k != expected_k
            or not active_cs
            or active_cs[0] != expected_c
            or len(set(active_cs)) != len(active_cs)
            or any(value not in tuple(float(item) for item in domain_c) for value in active_cs)
            or _loo_checkpoint_name(mode, active_k, active_cs)
            != active_raw.get("checkpoint_name")
        ):
            raise CheckpointValidationError("calibration grid active batch is not canonical")
        active_batch = {
            "k": active_k,
            "c_values": active_cs,
            "checkpoint_name": str(active_raw["checkpoint_name"]),
        }

    valid_attempts = [attempt for attempt in attempts if attempt.status == "valid"]
    winner_raw = state.get("winner")
    winner: dict[str, Any] | None = None
    winner_array = arrays.get("winner_loo_probabilities")
    if not valid_attempts:
        if winner_raw is not None or winner_array is not None:
            raise CheckpointValidationError("calibration grid has a winner without a valid attempt")
    else:
        if not isinstance(winner_raw, dict) or winner_array is None:
            raise CheckpointValidationError("calibration grid winner proof is incomplete")
        if winner_array.dtype != np.dtype(np.float64) or winner_array.shape != (
            pool.descriptors.shape[0],
        ):
            raise CheckpointValidationError("calibration grid winner probability array is invalid")
        if not np.all(np.isfinite(winner_array)) or np.any(winner_array <= 0.0):
            raise CheckpointValidationError("calibration grid winner probabilities are invalid")
        expected = valid_attempts[0]
        for attempt in valid_attempts[1:]:
            if (
                attempt.objective is not None
                and expected.objective is not None
                and attempt.objective < expected.objective
            ):
                expected = attempt
        try:
            winner_k = int(winner_raw["k"])
            winner_c = float(winner_raw["c"])
            winner_objective = float(winner_raw["objective"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CheckpointValidationError("calibration grid winner metadata is invalid") from exc
        if (
            winner_k != expected.k
            or winner_c != expected.c
            or winner_objective != expected.objective
            or winner_raw.get("bandwidth_fingerprint") != expected.bandwidth_fingerprint
            or winner_raw.get("probabilities_fingerprint") != _array_fingerprint(winner_array)
        ):
            raise CheckpointValidationError(
                "calibration grid winner is not the first canonical minimum"
            )
        winner = {
            "k": winner_k,
            "c": winner_c,
            "objective": winner_objective,
            "bandwidth_fingerprint": expected.bandwidth_fingerprint,
            "probabilities": winner_array,
        }
    elapsed = state.get("elapsed_wall_seconds", 0.0)
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise CheckpointValidationError("calibration grid elapsed walltime is invalid")
    retire_name = state.get("retire_checkpoint_name")
    if retire_name is not None and (
        not isinstance(retire_name, str)
        or not retire_name
        or Path(retire_name).name != retire_name
    ):
        raise CheckpointValidationError("calibration grid retirement handoff is invalid")
    raw_groups = state.get("c_batch_groups", [])
    if not isinstance(raw_groups, list) or any(not isinstance(item, dict) for item in raw_groups):
        raise CheckpointValidationError("calibration grid c-batch metadata is invalid")
    normalized_groups: list[dict[str, Any]] = []
    for item in raw_groups:
        values = item.get("c_values")
        if not isinstance(values, list) or not values:
            raise CheckpointValidationError("calibration grid c-batch values are invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise CheckpointValidationError("calibration grid c-batch values are invalid")
        normalized = dict(item)
        normalized["c_values"] = [float(value) for value in values]
        normalized_groups.append(normalized)
    return {
        "attempts": attempts,
        "pair_wall_seconds": pair_wall_seconds,
        "active_batch": active_batch,
        "winner": winner,
        "elapsed_wall_seconds": float(elapsed),
        "retire_checkpoint_name": retire_name,
        "c_batch_groups": normalized_groups,
    }


def _resolve_runtime_limits(
    settings: EntropyBandwidthSettings,
    pool: EntropyPool,
    resource_budget: ResourceBudgetService | None,
) -> tuple[EntropyBandwidthSettings, ResourceBudgetService]:
    """Fill operational limits from the shared budget without changing science."""

    service = resource_budget or build_resource_budget()
    remaining = service.remaining_managed_budget
    if remaining is None:
        raise ResourceCapacityError(
            service.unknown_memory_message("entropy bandwidth calibration"),
            operation="entropy bandwidth calibration",
        )
    if remaining < 1:
        raise ResourceCapacityError(
            "entropy bandwidth calibration has no allocatable runtime memory headroom",
            operation="entropy bandwidth calibration",
            requested_bytes=1,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    maximum_k = max(settings.k_candidates) if settings.mode == "automatic" else int(settings.k or 1)
    estimated_neighbour_bytes = int(
        pool.descriptors.nbytes
        + 16 * len(pool.descriptors) * maximum_k
        + 64 * len(pool.descriptors)
    )
    if estimated_neighbour_bytes > remaining:
        raise ResourceCapacityError(
            "entropy bandwidth calibration cannot reserve exact neighbour storage: "
            f"N={len(pool.descriptors)}, requested_bytes={estimated_neighbour_bytes}, "
            f"available_bytes={remaining}, k_max={maximum_k}",
            operation="entropy bandwidth calibration",
            requested_bytes=estimated_neighbour_bytes,
            available_bytes=remaining,
            reserved_headroom_bytes=service.budget.reserved_headroom_bytes,
        )
    work_bytes = min(
        settings.max_calibration_work_bytes or remaining,
        remaining,
    )
    radius_bytes = min(
        settings.max_radius_query_bytes or max(1, remaining // 2),
        work_bytes,
    )
    index_bytes = min(settings.max_index_bytes or remaining, remaining)
    resolved = replace(
        settings,
        max_neighbour_entries=settings.max_neighbour_entries,
        max_index_bytes=index_bytes,
        max_radius_query_bytes=radius_bytes,
        max_neighbour_cache_bytes=settings.max_neighbour_cache_bytes,
        max_calibration_work_bytes=work_bytes,
    )
    logger.info(
        "Entropy runtime budget admitted calibration: N=%d, remaining_bytes=%d, "
        "index_bytes=%d, radius_query_bytes=%d, work_bytes=%d, provenance=%s",
        len(pool.descriptors),
        remaining,
        index_bytes,
        radius_bytes,
        work_bytes,
        service.budget.provenance_of_budget,
    )
    return resolved, service


def calibrate_bandwidth(
    representation_or_pool: LocalEnvironmentRepresentation | EntropyPool | np.ndarray,
    settings: EntropyBandwidthSettings | Mapping[str, Any] | None = None,
    *,
    rows: Sequence[Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
    row_candidate_ids: Sequence[str] | None = None,
    transform_fingerprint: str | None = None,
    representation_fingerprint: str | None = None,
    mode: str | None = None,
    k: int | None = None,
    c: float | None = None,
    k_candidates: Sequence[int] | None = None,
    c_candidates: Sequence[float] | None = None,
    backend: str = DEFAULT_NEIGHBOUR_BACKEND_ID,
    metric: str = NEIGHBOUR_METRIC,
    chunk_size: int = 1024,
    max_neighbour_entries: int | None = None,
    max_index_bytes: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = None,
    max_calibration_work_bytes: int | None = None,
    calibration_batch_size: int | None = None,
    kernel_support_execution: str = "cpu",
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_gpu_support_bytes: int | None = None,
    max_gpu_neighbour_bytes: int | None = None,
    loo_checkpoint_every_sources: int | None = None,
    loo_checkpoint_every_seconds: float | None = None,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> BandwidthCalibrationResult:
    """Run deterministic manual or bounded-grid finite-pool calibration."""

    pool = (
        representation_or_pool
        if isinstance(representation_or_pool, EntropyPool)
        else build_entropy_pool(
            representation_or_pool,
            rows=rows,
            candidate_ids=candidate_ids,
            row_candidate_ids=row_candidate_ids,
            transform_fingerprint=transform_fingerprint,
            representation_fingerprint=representation_fingerprint,
        )
    )
    selected_settings = _settings_from_arguments(
        settings,
        mode=mode,
        k=k,
        c=c,
        k_candidates=k_candidates,
        c_candidates=c_candidates,
        backend=backend,
        metric=metric,
        chunk_size=chunk_size,
        max_neighbour_entries=max_neighbour_entries,
        max_index_bytes=max_index_bytes,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        max_calibration_work_bytes=max_calibration_work_bytes,
        calibration_batch_size=calibration_batch_size,
        kernel_support_execution=kernel_support_execution,
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_gpu_support_bytes=max_gpu_support_bytes,
        max_gpu_neighbour_bytes=max_gpu_neighbour_bytes,
        loo_checkpoint_every_sources=loo_checkpoint_every_sources,
        loo_checkpoint_every_seconds=loo_checkpoint_every_seconds,
    )
    selected_settings, runtime_budget = _resolve_runtime_limits(
        selected_settings,
        pool,
        resource_budget,
    )
    if pool.descriptors.shape[0] < 2:
        raise BandwidthCalibrationError("leave-one-out calibration requires at least two rows")

    index = build_neighbour_index(
        pool.descriptors,
        backend=selected_settings.backend,
        tile_size=selected_settings.chunk_size,
        max_index_bytes=selected_settings.max_index_bytes,
        max_radius_query_bytes=selected_settings.radius_query_bytes,
        max_neighbour_cache_bytes=selected_settings.max_neighbour_cache_bytes,
        gpu_device=selected_settings.gpu_device,
        gpu_devices=selected_settings.gpu_devices,
        gpu_tile_size=selected_settings.gpu_tile_size,
        max_gpu_neighbour_bytes=selected_settings.max_gpu_neighbour_bytes,
        resource_budget=runtime_budget,
    )

    if selected_settings.mode == "manual":
        if selected_settings.k is None or selected_settings.c is None:
            raise ValueError("manual bandwidth settings require explicit k and c")
        domain_k = (int(selected_settings.k),)
        domain_c = (float(selected_settings.c),)
    else:
        domain_k = selected_settings.k_candidates
        domain_c = selected_settings.c_candidates

    total_combinations = len(domain_k) * len(domain_c)
    calibration_started = time.perf_counter()
    grid_store: VersionedCheckpointStore | None = None
    grid_identity: dict[str, Any] | None = None
    restored_grid: dict[str, Any] | None = None
    if checkpoint_path is not None:
        grid_store = VersionedCheckpointStore(
            checkpoint_path,
            retention_generations=2,
            resource_budget=runtime_budget,
            retention_name=_CALIBRATION_GRID_CHECKPOINT_NAME,
        )
        grid_identity = _calibration_grid_identity(
            pool,
            selected_settings,
            domain_k,
            domain_c,
            index,
        )
        if not resume:
            grid_store.clear(_CALIBRATION_GRID_CHECKPOINT_NAME)
            grid_store.clean_stale(checkpoint_names={_CALIBRATION_GRID_CHECKPOINT_NAME})
        elif grid_store.exists(_CALIBRATION_GRID_CHECKPOINT_NAME):
            _stored_identity, stored_state, stored_arrays = grid_store.load(
                _CALIBRATION_GRID_CHECKPOINT_NAME,
                expected_identity=grid_identity,
            )
            restored_grid = _restore_calibration_grid_state(
                stored_state,
                stored_arrays,
                pool=pool,
                mode=selected_settings.mode,
                domain_k=domain_k,
                domain_c=domain_c,
            )
        elif grid_store.has_artifacts(_CALIBRATION_GRID_CHECKPOINT_NAME):
            raise CheckpointValidationError(
                "calibration grid checkpoint has retained data but no durable pointer"
            )
    operational_metrics: dict[str, Any] = {
        "wall_seconds": 0.0,
        "neighbour_query_seconds": 0.0,
        "loo_seconds": 0.0,
        "support_query_seconds": 0.0,
        "loo_pair_count": 0,
        "resource_lease_peak_bytes": 0,
        "scratch_peak_bytes": 0,
        "c_batch_group_count": 0,
        "c_batch_groups": [],
        "c_batch_reused_scans": 0,
        "c_batch_projected_savings": 0.0,
        "c_batch_pilot_seconds": 0.0,
        "c_batch_pilot_sources": 0,
        "c_batch_pilot_gpu_fp64_evaluations": 0,
        "c_batch_pilot_d2h_bytes": 0,
        "c_batch_actual_gpu_fp64_evaluations": 0,
        "c_batch_actual_d2h_bytes": 0,
        "c_batch_measured_scan_savings": 0,
        "c_batch_measured_d2h_savings_bytes": 0,
        "support_size_per_c": [],
    }
    if restored_grid is not None:
        restored_groups = list(restored_grid.get("c_batch_groups", []))
        operational_metrics["c_batch_groups"] = restored_groups
        operational_metrics["c_batch_group_count"] = len(restored_groups)
        operational_metrics["c_batch_reused_scans"] = sum(
            max(0, len(item.get("c_values", [])) - 1) for item in restored_groups
        )
        operational_metrics["c_batch_projected_savings"] = sum(
            float(item.get("projected_savings", 0.0) or 0.0) for item in restored_groups
        )

    def sample_resource_leases() -> None:
        operational_metrics["resource_lease_peak_bytes"] = max(
            int(operational_metrics["resource_lease_peak_bytes"]),
            int(runtime_budget.reserved_memory_bytes),
        )
        operational_metrics["scratch_peak_bytes"] = max(
            int(operational_metrics["scratch_peak_bytes"]),
            int(runtime_budget.reserved_scratch_bytes),
        )

    logger.info(
        "Bandwidth calibration started: mode=%s, ordered combinations=%d",
        selected_settings.mode,
        total_combinations,
    )
    if pool.descriptors.shape[0] < 2:
        elapsed, _ = _progress_timing(0, total_combinations, calibration_started)
        logger.error(
            "Bandwidth calibration failed: no valid pair, attempted=0, valid=0, "
            "invalid=0, elapsed=%.3fs",
            elapsed,
        )
        raise BandwidthCalibrationError("leave-one-out calibration requires at least two rows")

    attempts: list[CalibrationAttempt] = (
        [] if restored_grid is None else list(restored_grid["attempts"])
    )
    pair_wall_seconds: list[float] = (
        [] if restored_grid is None else list(restored_grid["pair_wall_seconds"])
    )
    pairs_reused = len(attempts)
    pairs_computed = 0
    grid_completed_commits = 0
    resumed_elapsed_wall_seconds = (
        0.0 if restored_grid is None else float(restored_grid["elapsed_wall_seconds"])
    )
    restored_retire_checkpoint_name = (
        None if restored_grid is None else restored_grid["retire_checkpoint_name"]
    )
    active_grid_batch = None if restored_grid is None else restored_grid["active_batch"]
    active_grid_batch_consumed = active_grid_batch is None
    radii_by_k: dict[int, ExactNeighbourResult | str] = {}
    if selected_settings.mode == "automatic" and index is not None:
        indexed_orders: list[int] = []
        location_limit = int(index.unique_locations.shape[0] - 1)
        for candidate_k in domain_k:
            if candidate_k > location_limit:
                radii_by_k[candidate_k] = (
                    f"k={candidate_k} requires at least {candidate_k} other distinct "
                    f"descriptor locations; the pool contains only {location_limit}"
                )
            elif (
                selected_settings.max_neighbour_entries is not None
                and pool.descriptors.shape[0] * candidate_k
                > selected_settings.max_neighbour_entries
            ):
                radii_by_k[candidate_k] = (
                    "indexed exact neighbour result exceeds its explicit bounded-memory "
                    f"limit: N={pool.descriptors.shape[0]}, "
                    f"required_entries={pool.descriptors.shape[0] * candidate_k}, "
                    f"max_entries={selected_settings.max_neighbour_entries}"
                )
            else:
                indexed_orders.append(candidate_k)
        if indexed_orders:
            common_neighbour_options: dict[str, Any] = {
                "chunk_size": selected_settings.chunk_size,
                "max_neighbour_entries": selected_settings.max_neighbour_entries,
                "representation_fingerprint": pool.representation_fingerprint,
                "row_ids": _pool_row_ids(pool),
                "resource_budget": runtime_budget,
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": "knn",
                "resume": resume,
            }
            neighbour_started = time.perf_counter()
            if isinstance(index, ExactTiledGPUNeighbourIndex):
                radii_by_k.update(
                    compute_gpu_tiled_neighbours_many(
                        pool.descriptors,
                        indexed_orders,
                        index=index,
                        **common_neighbour_options,
                    )
                )
            elif isinstance(index, TiledCPUNeighbourIndex):
                radii_by_k.update(
                    compute_tiled_cpu_neighbours_many(
                        pool.descriptors,
                        indexed_orders,
                        index=index,
                        **common_neighbour_options,
                    )
                )
            elif isinstance(index, IndexedCPUNeighbourIndex):
                radii_by_k.update(
                    compute_indexed_cpu_neighbours_many(
                        pool.descriptors,
                        indexed_orders,
                        index=index,
                        **common_neighbour_options,
                    )
                )
            else:
                raise RuntimeError(
                    "automatic entropy calibration has no compatible neighbour index"
                )
            operational_metrics["neighbour_query_seconds"] += max(
                0.0, time.perf_counter() - neighbour_started
            )
            sample_resource_leases()

    winner: tuple[float, FrozenBandwidths, np.ndarray] | None = None
    restored_winner = None if restored_grid is None else restored_grid["winner"]
    if restored_winner is not None and selected_settings.mode == "automatic":
        winner_k = int(restored_winner["k"])
        restored_neighbours = radii_by_k.get(winner_k)
        if not isinstance(restored_neighbours, ExactNeighbourResult):
            raise CheckpointValidationError(
                "calibration grid winner cannot be reconstructed from current neighbour results"
            )
        restored_frozen = _freeze_from_neighbours(
            pool,
            restored_neighbours,
            float(restored_winner["c"]),
        )
        if restored_frozen.fingerprint != restored_winner["bandwidth_fingerprint"]:
            raise CheckpointValidationError(
                "calibration grid winner bandwidth fingerprint does not match current neighbours"
            )
        restored_probabilities = np.asarray(restored_winner["probabilities"], dtype=np.float64)
        if restored_probabilities.shape != (pool.descriptors.shape[0],):
            raise CheckpointValidationError(
                "calibration grid winner probabilities have the wrong shape"
            )
        winner = (
            float(restored_winner["objective"]),
            restored_frozen,
            restored_probabilities,
        )
    completed_combinations = len(attempts)
    valid_count = sum(attempt.status == "valid" for attempt in attempts)
    invalid_count = sum(attempt.status == "invalid" for attempt in attempts)

    def record_attempt(attempt: CalibrationAttempt) -> None:
        nonlocal completed_combinations, valid_count, invalid_count, pairs_computed
        if len(attempts) >= total_combinations:
            raise CheckpointValidationError("calibration grid received an extra attempt")
        expected_position = len(attempts)
        expected_pair = (
            int(domain_k[expected_position // len(domain_c)]),
            float(domain_c[expected_position % len(domain_c)]),
        )
        if (int(attempt.k), float(attempt.c)) != expected_pair:
            raise CheckpointValidationError(
                "calibration grid attempt is not the first unfinished canonical pair"
            )
        attempts.append(attempt)
        pairs_computed += 1
        completed_combinations += 1
        if attempt.status == "valid":
            valid_count += 1
        else:
            invalid_count += 1
        elapsed, remaining = _progress_timing(
            completed_combinations,
            total_combinations,
            calibration_started,
        )
        percentage = 100.0 * completed_combinations / total_combinations
        progress_values = (
            attempt.k,
            attempt.c,
            completed_combinations,
            total_combinations,
            percentage,
            valid_count,
            invalid_count,
            elapsed,
            remaining,
        )
        if attempt.status == "valid":
            logger.info(
                "Bandwidth calibration pair completed: k=%d, c=%g, J(k,c)=%.12g, "
                "%d/%d combinations (%.1f%%), valid=%d, invalid=%d, elapsed=%.3fs, "
                "estimated remaining=%.3fs, reused=false, evaluated=true",
                *progress_values[:2],
                attempt.objective,
                *progress_values[2:],
            )
        else:
            logger.info(
                "Bandwidth calibration pair completed: k=%d, c=%g, invalid=%s, "
                "%d/%d combinations (%.1f%%), valid=%d, invalid=%d, elapsed=%.3fs, "
                "estimated remaining=%.3fs, reused=false, evaluated=true",
                *progress_values[:2],
                attempt.reason or "unspecified invalidity",
                *progress_values[2:],
            )

    def _grid_winner_payload() -> dict[str, Any] | None:
        if winner is None:
            return None
        objective_value, frozen, _probabilities = winner
        matching = next(
            (
                attempt
                for attempt in attempts
                if attempt.status == "valid"
                and attempt.k == frozen.k
                and attempt.c == frozen.c
                and attempt.objective == objective_value
            ),
            None,
        )
        if matching is None or matching.bandwidth_fingerprint != frozen.fingerprint:
            raise CheckpointValidationError("calibration grid winner has no matching attempt proof")
        return {
            "k": frozen.k,
            "c": frozen.c,
            "objective": float(objective_value),
            "bandwidth_fingerprint": frozen.fingerprint,
            "probabilities_fingerprint": _array_fingerprint(_probabilities),
        }

    def publish_grid(
        active_batch: Mapping[str, Any] | None,
        *,
        retire_checkpoint_name: str | None = None,
    ) -> None:
        if grid_store is None or grid_identity is None:
            return
        winner_payload = _grid_winner_payload()
        state: dict[str, Any] = {
            "schema": _CALIBRATION_GRID_SCHEMA,
            "cursor": len(attempts),
            "attempts": [_attempt_payload(attempt) for attempt in attempts],
            "pair_wall_seconds": [float(value) for value in pair_wall_seconds],
            "c_batch_groups": list(operational_metrics.get("c_batch_groups", [])),
            "active_batch": None if active_batch is None else dict(active_batch),
            "retire_checkpoint_name": retire_checkpoint_name,
            "winner": winner_payload,
            "elapsed_wall_seconds": resumed_elapsed_wall_seconds
            + max(0.0, time.perf_counter() - calibration_started),
        }
        arrays: dict[str, np.ndarray] = {}
        if winner is not None:
            arrays["winner_loo_probabilities"] = np.asarray(winner[2], dtype=np.float64)
        grid_store.write(
            _CALIBRATION_GRID_CHECKPOINT_NAME,
            identity=grid_identity,
            state=state,
            arrays=arrays,
        )

    loo_cleanup_store: VersionedCheckpointStore | None = None

    def retire_loo_checkpoint(checkpoint_name: str) -> None:
        nonlocal loo_cleanup_store
        if checkpoint_path is None:
            return
        if loo_cleanup_store is None:
            loo_cleanup_store = VersionedCheckpointStore(
                checkpoint_path,
                resource_budget=runtime_budget,
            )
        loo_cleanup_store.clear(checkpoint_name)
        loo_cleanup_store.clean_stale(checkpoint_names={checkpoint_name})

    if restored_retire_checkpoint_name is not None:
        retire_loo_checkpoint(restored_retire_checkpoint_name)

    def append_pair_timings(start_position: int, started: float) -> None:
        added = len(attempts) - start_position
        if added < 0:
            raise CheckpointValidationError("calibration grid attempt cursor moved backwards")
        if added:
            per_pair = max(0.0, time.perf_counter() - started) / added
            pair_wall_seconds.extend([per_pair] * added)

    def source_progress_logger(
        candidate_k: int,
        candidate_cs: Sequence[float],
    ) -> Callable[[int, int], None]:
        ordered_cs = tuple(float(value) for value in candidate_cs)
        c_label = (
            f"c={ordered_cs[0]:g}"
            if len(ordered_cs) == 1
            else "c_batch=" + ",".join(f"{value:g}" for value in ordered_cs)
        )
        pair_started = time.perf_counter()
        next_percent = 5
        last_completed = 0
        last_log_time = pair_started

        def report(completed: int, total_sources: int) -> None:
            nonlocal next_percent, last_completed, last_log_time
            if completed <= last_completed:
                return
            last_completed = completed
            now = time.perf_counter()
            if total_sources < 20:
                should_report = True
            else:
                should_report = (
                    completed * 100 >= next_percent * total_sources
                    or now - last_log_time >= 10.0
                    or completed == total_sources
                )
            if not should_report:
                return
            elapsed, remaining = _progress_timing(completed, total_sources, pair_started)
            percentage = 100.0 * completed / total_sources
            active = operational_metrics.get("loo_active_support_metrics", {})
            metrics: Mapping[str, Any] = active if isinstance(active, Mapping) else {}

            def metric(name: str, default: int | float | str = "unavailable") -> Any:
                value = metrics.get(name)
                return default if value is None else value

            stats_status = str(metrics.get("support_provider_stats_status", "not available"))
            current_support = metric("current_support_count")
            radius_queries = metric("support_provider_radius_queries")

            logger.info(
                "Bandwidth calibration pair source progress: k=%d, %s, source=%d/%d "
                "(%.1f%%), phase=radius-support+loo, support_provider=%s, "
                "support_metrics=%s, current_support=%s, "
                "min_support=%s, mean_support=%s, "
                "max_support=%s, "
                "radius_queries=%s, unique_distance_evaluations=%s, "
                "atomic_distance_evaluations=%s, "
                "support_cache_bytes=0, "
                "cache_hit=%s, cache_miss=%s, cache_eviction=%s, "
                "cache_admission_rejected=%s, cache_bytes=%s, cache_peak_bytes=%s, "
                "cache_time_saved_seconds=%s, "
                "support_cache_peak_bytes=0, grouping_bytes=%s, grouping_build_seconds=%s, "
                "tree_build_seconds=%s, index_bytes=%s, query_workspace_peak_bytes=%s, "
                "process_peak_rss_bytes=%s, cgroup_memory_current_bytes=%s, "
                "elapsed=%.3fs, estimated remaining=%.3fs",
                candidate_k,
                c_label,
                completed,
                total_sources,
                percentage,
                str(metrics.get("support_provider", "not available")),
                stats_status,
                str(current_support),
                str(metrics.get("min_support_count", "unavailable")),
                str(metrics.get("mean_support_count", "unavailable")),
                str(metrics.get("max_support_count", "unavailable")),
                str(radius_queries),
                str(metric("support_provider_unique_distance_evaluations")),
                str(metric("support_provider_atomic_distance_evaluations")),
                str(metrics.get("cache_hit", "unavailable")),
                str(metrics.get("cache_miss", "unavailable")),
                str(metrics.get("cache_eviction", "unavailable")),
                str(metrics.get("cache_admission_rejected", "unavailable")),
                str(metrics.get("cache_bytes", "unavailable")),
                str(metrics.get("cache_peak_bytes", "unavailable")),
                str(metrics.get("cache_time_saved_seconds", "unavailable")),
                str(metrics.get("grouping_bytes", "unavailable")),
                str(metrics.get("grouping_build_seconds", "unavailable")),
                str(metrics.get("tree_build_seconds", "unavailable")),
                str(metrics.get("index_bytes", "unavailable")),
                str(metrics.get("query_workspace_peak_bytes", "unavailable")),
                str(metrics.get("process_peak_rss_bytes", "unavailable")),
                str(metrics.get("cgroup_memory_current_bytes", "unavailable")),
                elapsed,
                remaining,
            )
            last_log_time = now
            if total_sources >= 20:
                while next_percent <= 100 and completed * 100 >= next_percent * total_sources:
                    next_percent += 5

        return report

    for reused_attempt in attempts:
        logger.info(
            "Bandwidth calibration pair reused: k=%d, c=%g, status=%s, objective=%s, "
            "reused=true, evaluated=false",
            reused_attempt.k,
            reused_attempt.c,
            reused_attempt.status,
            reused_attempt.objective,
        )

    def c_batches(
        neighbours: ExactNeighbourResult,
        candidate_cs: Sequence[float] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        ordered = tuple(
            float(value) for value in (domain_c if candidate_cs is None else candidate_cs)
        )
        if selected_settings.mode == "manual":
            return (ordered,)
        row_count = int(pool.descriptors.shape[0])
        per_c_bytes = 16 * row_count
        source_workspace_bound = _estimate_radius_workspace_bytes(
            row_count,
            row_count,
            int(pool.descriptors.shape[1]),
            row_count,
        )
        available = selected_settings.operational_work_bytes - 8192 - source_workspace_bound
        maximum = available // max(1, per_c_bytes)
        if maximum < 1:
            raise RadiusQueryCapacityError(
                "batched calibration workspace cannot fit one c value: "
                f"N={pool.descriptors.shape[0]}, d'={pool.descriptors.shape[1]}, "
                f"requested_bytes={per_c_bytes + 8192 + source_workspace_bound}, "
                f"max_calibration_work_bytes={selected_settings.operational_work_bytes}, "
                f"k_domain={tuple(int(value) for value in domain_k)}"
            )
        width = min(
            len(ordered),
            int(maximum),
            (
                int(selected_settings.calibration_batch_size)
                if selected_settings.calibration_batch_size is not None
                else len(ordered)
            ),
        )
        predicted_support: dict[float, float | None] = {value: None for value in ordered}
        pilot_seconds = 0.0
        pilot_gpu_metrics: Mapping[str, Any] = {}
        pilot_sample_count = 0
        if width > 1 and selected_settings.kernel_support_execution == "exact_tiled_gpu":
            pilot_started = time.perf_counter()
            pilot_provider = ExactTiledGPUSupport(
                pool.descriptors,
                device=selected_settings.gpu_device,
                gpu_devices=selected_settings.gpu_devices,
                tile_size=selected_settings.gpu_tile_size,
                max_device_bytes=selected_settings.max_gpu_support_bytes,
                max_host_bytes=selected_settings.radius_query_bytes,
                resource_budget=runtime_budget,
                capacity_context=f"c-batch GPU pilot k={neighbours.k}",
            )
            try:
                pilot_source_count = min(_C_BATCH_PILOT_SOURCE_LIMIT, row_count)
                if pilot_source_count < 1:
                    raise RadiusQueryCapacityError(
                        "GPU c-batch pilot cannot fit one source in its bounded workspace"
                    )
                sample_rows = np.unique(
                    np.linspace(0, row_count - 1, pilot_source_count, dtype=np.int64)
                )
                pilot_sample_count = int(sample_rows.size)
                pilot_support_totals = np.zeros(len(ordered), dtype=np.float64)
                pilot_source_tile = max(1, int(pilot_provider.source_tile_size))
                for pilot_start in range(0, sample_rows.size, pilot_source_tile):
                    pilot_rows = sample_rows[pilot_start : pilot_start + pilot_source_tile]
                    pilot_radii = np.stack(
                        [candidate_c * neighbours.radii[pilot_rows] for candidate_c in ordered]
                    )
                    pilot_offsets, _pilot_targets, pilot_distances = (
                        pilot_provider.support_many(pilot_rows, pilot_radii)
                    )
                    for candidate_index in range(len(ordered)):
                        for source_offset in range(pilot_rows.size):
                            begin = int(pilot_offsets[source_offset])
                            end = int(pilot_offsets[source_offset + 1])
                            pilot_support_totals[candidate_index] += np.count_nonzero(
                                pilot_distances[begin:end]
                                < pilot_radii[candidate_index, source_offset]
                            )
                for candidate_index, candidate_c in enumerate(ordered):
                    predicted_support[candidate_c] = (
                        pilot_support_totals[candidate_index] / sample_rows.size
                    )
                pilot_gpu_metrics = pilot_provider.operational_metrics()
            finally:
                pilot_seconds = max(0.0, time.perf_counter() - pilot_started)
                pilot_provider.close()
            operational_metrics["c_batch_pilot_seconds"] = operational_metrics.get(
                "c_batch_pilot_seconds", 0.0
            ) + pilot_seconds
            operational_metrics["c_batch_pilot_sources"] = operational_metrics.get(
                "c_batch_pilot_sources", 0
            ) + int(sample_rows.size)
            operational_metrics["c_batch_pilot_gpu_fp64_evaluations"] = operational_metrics.get(
                "c_batch_pilot_gpu_fp64_evaluations", 0
            ) + int(pilot_gpu_metrics.get("gpu_fp64_evaluations", 0) or 0)
            operational_metrics["c_batch_pilot_d2h_bytes"] = operational_metrics.get(
                "c_batch_pilot_d2h_bytes", 0
            ) + int(pilot_gpu_metrics.get("gpu_d2h_bytes", 0) or 0)
            logger.info(
                "Bandwidth GPU c-batch pilot completed: version=%s, k=%d, sources=%d, "
                "seconds=%.6f, gpu_fp64_evaluations=%d, d2h_bytes=%d, predicted_support=%s",
                _C_BATCH_PILOT_VERSION,
                neighbours.k,
                sample_rows.size,
                pilot_seconds,
                int(pilot_gpu_metrics.get("gpu_fp64_evaluations", 0) or 0),
                int(pilot_gpu_metrics.get("gpu_d2h_bytes", 0) or 0),
                ", ".join(f"c={value:g}:{predicted_support[value]}" for value in ordered),
            )
        elif isinstance(index, IndexedCPUNeighbourIndex):
            row_count = int(pool.descriptors.shape[0])
            sample_count = min(16, row_count)
            pilot_cost = sample_count * len(ordered)
            maximum_saved_queries = max(
                0,
                (len(ordered) - (len(ordered) + width - 1) // width) * row_count,
            )
            if width > 1 and pilot_cost < maximum_saved_queries:
                sample_rows = np.unique(np.linspace(0, row_count - 1, sample_count, dtype=np.int64))
                logger.info(
                    "Bandwidth c-batch pilot started: version=%s, k=%d, samples=%d, "
                    "c_count=%d, estimated_pilot_queries=%d, maximum_saved_queries=%d",
                    _C_BATCH_PILOT_VERSION,
                    neighbours.k,
                    sample_rows.size,
                    len(ordered),
                    pilot_cost,
                    maximum_saved_queries,
                )
                for candidate_c in ordered:
                    support_total = 0
                    successful = 0
                    try:
                        for source_row in sample_rows:
                            radius = float(candidate_c * neighbours.radii[int(source_row)])
                            targets, _distances = index.radius_support(
                                int(source_row),
                                radius,
                                context=(
                                    f"c-batch pilot k={neighbours.k}, c={candidate_c:g}, "
                                    f"source_row={int(source_row)}"
                                ),
                            )
                            support_total += int(targets.shape[0])
                            successful += 1
                            index.pilot_support_queries += 1
                            index.pilot_support_count_total += int(targets.shape[0])
                            index.pilot_support_samples += 1
                    except (FloatingPointError, OverflowError, ValueError, MemoryError) as exc:
                        logger.info(
                            "Bandwidth c-batch pilot unavailable: k=%d, c=%g, reason=%s",
                            neighbours.k,
                            candidate_c,
                            exc,
                        )
                        predicted_support[candidate_c] = None
                    else:
                        if successful:
                            predicted_support[candidate_c] = support_total / successful
                logger.info(
                    "Bandwidth c-batch pilot completed: version=%s, k=%d, predicted_support=%s",
                    _C_BATCH_PILOT_VERSION,
                    neighbours.k,
                    ", ".join(f"c={value:g}:{predicted_support[value]}" for value in ordered),
                )
            else:
                logger.info(
                    "Bandwidth c-batch pilot skipped: version=%s, k=%d, "
                    "estimated_pilot_queries=%d, maximum_saved_queries=%d",
                    _C_BATCH_PILOT_VERSION,
                    neighbours.k,
                    pilot_cost,
                    maximum_saved_queries,
                )
        elif selected_settings.kernel_support_execution == "exact_tiled_gpu":
            logger.info(
                "Bandwidth GPU c-batch pilot skipped: version=%s, k=%d, "
                "batch_width=%d does not admit a shared group",
                _C_BATCH_PILOT_VERSION,
                neighbours.k,
                width,
            )

        is_gpu = selected_settings.kernel_support_execution == "exact_tiled_gpu"
        scan_pairs = float(row_count * row_count)
        scan_seconds = 0.0
        d2h_seconds = 0.0
        cpu_verification_seconds = 0.0
        cpu_rescanned_count = 0
        if is_gpu and pilot_gpu_metrics:
            pilot_source_count = max(1, pilot_sample_count)
            scale_to_pool = row_count / pilot_source_count
            scan_pairs = max(
                1.0,
                float(pilot_gpu_metrics.get("gpu_fp64_evaluations", 0) or 0)
                * scale_to_pool,
            )
            scan_seconds = float(pilot_gpu_metrics.get("gpu_seconds", 0.0) or 0.0) * scale_to_pool
            d2h_seconds = (
                float(pilot_gpu_metrics.get("gpu_d2h_seconds", 0.0) or 0.0) * scale_to_pool
            )
            cpu_verification_seconds = (
                float(pilot_gpu_metrics.get("gpu_cpu_verification_seconds", 0.0) or 0.0)
                * scale_to_pool
            )
            cpu_rescanned_count = int(pilot_gpu_metrics.get("gpu_cpu_rescanned_count", 0) or 0)
        scan_time_per_pair = (
            scan_seconds / scan_pairs if scan_seconds > 0.0 else 1.0e-9
        )
        verification_time_per_candidate = (
            cpu_verification_seconds / max(1, cpu_rescanned_count)
            if cpu_verification_seconds > 0.0
            else 1.0e-9
        )

        def estimated_cost(batch: Sequence[float]) -> float:
            supports = [predicted_support.get(value) for value in batch]
            if any(value is None for value in supports):
                return float("inf")
            numeric_supports = [float(value) for value in supports if value is not None]
            candidate_count = max(numeric_supports, default=0.0) * row_count
            candidate_verification = candidate_count * verification_time_per_candidate
            normalization = (
                sum(numeric_supports)
                * row_count
                * _C_BATCH_NORMALIZATION_COST
            )
            scan = scan_pairs * scan_time_per_pair
            if d2h_seconds > 0.0:
                scan += d2h_seconds
            # The C-by-N numerator and the bounded CSR candidate output are
            # deliberately part of the model, so grouping is not based only
            # on a support-size ratio.
            memory_bytes = (
                16.0 * len(batch) * row_count
                + 32.0 * max(numeric_supports, default=0.0) * min(row_count, width)
            )
            memory_seconds = memory_bytes / _C_BATCH_MEMORY_BANDWIDTH_BYTES_PER_SECOND
            checkpoint_seconds = _C_BATCH_CHECKPOINT_COST_SECONDS
            return (
                scan
                + candidate_verification
                + normalization
                + memory_seconds
                + checkpoint_seconds
            )

        def workspace_admissible(batch: Sequence[float]) -> bool:
            maximum_support = max(
                (predicted_support.get(value) or 0.0 for value in batch),
                default=0.0,
            )
            query_sources = min(row_count, int(selected_settings.chunk_size))
            requested = (
                8192
                + int(source_workspace_bound)
                + 16 * len(batch) * row_count
                + 32 * int(math.ceil(maximum_support)) * query_sources
            )
            return requested <= int(selected_settings.operational_work_bytes)

        # Preserve configured c order. Adjacent candidates share one support
        # scan only when the measured/estimated cost and bounded workspace
        # justify the larger canonical verification superset.
        grouped: list[tuple[float, ...]] = []
        current: list[float] = []
        for candidate_c in ordered:
            if not current:
                current.append(candidate_c)
                continue
            current_min = predicted_support.get(current[0])
            candidate_estimate = predicted_support.get(candidate_c)
            comparable = (
                current_min is not None
                and candidate_estimate is not None
                and candidate_estimate / max(current_min, 1.0) <= _C_BATCH_SUPPORT_RATIO_LIMIT
            )
            proposed = (*current, candidate_c)
            singleton_cost = sum(estimated_cost((value,)) for value in proposed)
            shared_cost = estimated_cost(proposed)
            beneficial = shared_cost < singleton_cost
            admissible = workspace_admissible(proposed)
            if len(current) < width and comparable and beneficial and admissible:
                current.append(candidate_c)
            else:
                grouped.append(tuple(current))
                current = [candidate_c]
        if current:
            grouped.append(tuple(current))
        batches = tuple(grouped)
        decision_records: list[dict[str, Any]] = []
        for batch in batches:
            singleton_cost = sum(estimated_cost((value,)) for value in batch)
            shared_cost = estimated_cost(batch)
            projected_savings = (
                max(0.0, singleton_cost - shared_cost)
                if math.isfinite(singleton_cost) and math.isfinite(shared_cost)
                else None
            )
            decision_records.append(
                {
                    "k": int(neighbours.k),
                    "c_values": list(batch),
                    "predicted_support": [predicted_support.get(value) for value in batch],
                    "estimated_cost": shared_cost if math.isfinite(shared_cost) else None,
                    "singleton_cost": (
                        singleton_cost if math.isfinite(singleton_cost) else None
                    ),
                    "projected_savings": projected_savings,
                    "choice": "shared" if len(batch) > 1 else "singleton",
                    "pilot_seconds": pilot_seconds,
                    "pilot_gpu_fp64_evaluations": int(
                        pilot_gpu_metrics.get("gpu_fp64_evaluations", 0) or 0
                    ),
                    "pilot_gpu_launch_count": int(
                        pilot_gpu_metrics.get("gpu_launch_count", 0) or 0
                    ),
                    "pilot_d2h_bytes": int(pilot_gpu_metrics.get("gpu_d2h_bytes", 0) or 0),
                    "gpu": is_gpu,
                }
            )
        operational_metrics.setdefault("c_batch_groups", []).extend(decision_records)
        operational_metrics["c_batch_group_count"] = operational_metrics.get(
            "c_batch_group_count", 0
        ) + len(batches)
        operational_metrics["c_batch_reused_scans"] = operational_metrics.get(
            "c_batch_reused_scans", 0
        ) + sum(max(0, len(batch) - 1) for batch in batches)
        operational_metrics["c_batch_projected_savings"] = operational_metrics.get(
            "c_batch_projected_savings", 0.0
        ) + sum(
            float(record["projected_savings"] or 0.0) for record in decision_records
        )
        logger.info(
            "Bandwidth calibration c batching: version=%s, k=%d, batch_size=%d, c_count=%d, "
            "c_batches=%d, estimated_batch_workspace_bytes=%d, max_calibration_work_bytes=%d, "
            "pilot_seconds=%.6f, gpu=%s",
            _C_BATCH_PILOT_VERSION,
            neighbours.k,
            width,
            len(ordered),
            len(batches),
            max(8192 + len(batch) * per_c_bytes + source_workspace_bound for batch in batches),
            selected_settings.operational_work_bytes,
            pilot_seconds,
            is_gpu,
        )
        for batch in batches:
            logger.info(
                "Bandwidth calibration c-batch planned: k=%d, c_batch=%s, predicted_support=%s, "
                "choice=%s, estimated_cost=%.6g, singleton_cost=%.6g, projected_savings=%.6g",
                neighbours.k,
                ",".join(f"{value:g}" for value in batch),
                ",".join(str(predicted_support.get(value)) for value in batch),
                "shared" if len(batch) > 1 else "singleton",
                estimated_cost(batch),
                sum(estimated_cost((value,)) for value in batch),
                max(0.0, sum(estimated_cost((value,)) for value in batch) - estimated_cost(batch)),
            )
        return batches

    for candidate_k in domain_k:
        completed_for_k = {
            attempt.c
            for attempt in attempts
            if attempt.k == candidate_k
        }
        pending_for_k = tuple(
            float(candidate_c)
            for candidate_c in domain_c
            if float(candidate_c) not in completed_for_k
        )
        if not pending_for_k and not (
            restored_winner is not None
            and winner is None
            and int(restored_winner["k"]) == int(candidate_k)
        ):
            continue
        try:
            neighbours = radii_by_k.get(candidate_k)
            if neighbours is None:
                neighbours = compute_neighbours(
                    pool.descriptors,
                    candidate_k,
                    backend=selected_settings.backend,
                    index=index,
                    chunk_size=selected_settings.chunk_size,
                    max_neighbour_entries=selected_settings.max_neighbour_entries,
                    representation_fingerprint=pool.representation_fingerprint,
                    row_ids=_pool_row_ids(pool),
                    resource_budget=runtime_budget,
                    checkpoint_path=checkpoint_path,
                    checkpoint_name=f"knn-k{candidate_k}",
                    resume=resume,
                )
                radii_by_k[candidate_k] = neighbours
            if isinstance(neighbours, str):
                raise ValueError(neighbours)
        except (ValueError, TypeError) as exc:
            reason = str(exc)
            radii_by_k[candidate_k] = reason
            started = time.perf_counter()
            start_position = len(attempts)
            for candidate_c in pending_for_k:
                record_attempt(
                    CalibrationAttempt(candidate_k, float(candidate_c), "invalid", reason=reason)
                )
            append_pair_timings(start_position, started)
            publish_grid(None)
            continue

        if (
            restored_winner is not None
            and winner is None
            and int(restored_winner["k"]) == int(candidate_k)
        ):
            restored_frozen = _freeze_from_neighbours(
                pool,
                neighbours,
                float(restored_winner["c"]),
            )
            if restored_frozen.fingerprint != restored_winner["bandwidth_fingerprint"]:
                raise CheckpointValidationError(
                    "calibration grid winner bandwidth fingerprint does not match "
                    "current neighbours"
                )
            winner = (
                float(restored_winner["objective"]),
                restored_frozen,
                np.asarray(restored_winner["probabilities"], dtype=np.float64),
            )

        if not pending_for_k:
            continue

        planned_batches = c_batches(neighbours, pending_for_k)
        if (
            active_grid_batch is not None
            and not active_grid_batch_consumed
            and int(active_grid_batch["k"]) == int(candidate_k)
        ):
            active_values = tuple(float(value) for value in active_grid_batch["c_values"])
            remaining_planned: list[tuple[float, ...]] = []
            for planned in planned_batches:
                remainder = tuple(value for value in planned if value not in active_values)
                if remainder:
                    remaining_planned.append(remainder)
            batches = (active_values, *remaining_planned)
            active_grid_batch_consumed = True
        else:
            batches = planned_batches
        for planned_batch in batches:
            c_batch = tuple(
                float(candidate_c)
                for candidate_c in planned_batch
                if float(candidate_c) not in completed_for_k
            )
            if not c_batch:
                continue
            logger.info(
                "Bandwidth calibration c-batch started: k=%d, c_batch=%s",
                candidate_k,
                ",".join(f"{value:g}" for value in c_batch),
            )
            loo_checkpoint_name = _loo_checkpoint_name(
                selected_settings.mode,
                candidate_k,
                c_batch,
            )
            pair_started = time.perf_counter()
            start_position = len(attempts)
            frozen_batch: tuple[FrozenBandwidths, ...] = ()
            try:
                frozen_batch = tuple(
                    _freeze_from_neighbours(pool, neighbours, candidate_c)
                    for candidate_c in c_batch
                )
                publish_grid(
                    {
                        "k": int(candidate_k),
                        "c_values": list(c_batch),
                        "checkpoint_name": loo_checkpoint_name,
                    }
                )
                loo_started = time.perf_counter()
                if selected_settings.mode == "manual":
                    frozen = frozen_batch[0]
                    objectives = (
                        evaluate_leave_one_out_objective(
                            pool.descriptors,
                            pool.probabilities,
                            frozen.bandwidths,
                            chunk_size=selected_settings.chunk_size,
                            progress_callback=source_progress_logger(candidate_k, c_batch),
                            backend=selected_settings.backend,
                            index=index,
                            max_radius_query_bytes=selected_settings.radius_query_bytes,
                            resource_budget=runtime_budget,
                            capacity_context=f"k={candidate_k}, c={c_batch[0]:g}",
                            checkpoint_path=checkpoint_path,
                            checkpoint_name=loo_checkpoint_name,
                            checkpoint_every_sources=selected_settings.loo_checkpoint_every_sources,
                            checkpoint_every_seconds=selected_settings.loo_checkpoint_every_seconds,
                            resume=resume,
                            preserve_checkpoint=grid_store is not None,
                            diagnostics=operational_metrics,
                            kernel_support_execution=selected_settings.kernel_support_execution,
                            gpu_device=selected_settings.gpu_device,
                            gpu_devices=selected_settings.gpu_devices,
                            gpu_tile_size=selected_settings.gpu_tile_size,
                            max_gpu_support_bytes=selected_settings.max_gpu_support_bytes,
                        ),
                    )
                else:
                    objectives = evaluate_leave_one_out_objectives(
                        pool.descriptors,
                        pool.probabilities,
                        np.stack([frozen.bandwidths for frozen in frozen_batch]),
                        chunk_size=selected_settings.chunk_size,
                        progress_callback=source_progress_logger(candidate_k, c_batch),
                        backend=selected_settings.backend,
                        index=index,
                        max_radius_query_bytes=selected_settings.radius_query_bytes,
                        max_calibration_work_bytes=selected_settings.operational_work_bytes,
                        resource_budget=runtime_budget,
                        capacity_context=(
                            f"k={candidate_k}, c_batch="
                            + ",".join(f"{value:g}" for value in c_batch)
                        ),
                        checkpoint_path=checkpoint_path,
                        checkpoint_name=loo_checkpoint_name,
                        checkpoint_every_sources=selected_settings.loo_checkpoint_every_sources,
                        checkpoint_every_seconds=selected_settings.loo_checkpoint_every_seconds,
                        resume=resume,
                        preserve_checkpoint=grid_store is not None,
                        diagnostics=operational_metrics,
                        kernel_support_execution=selected_settings.kernel_support_execution,
                        gpu_device=selected_settings.gpu_device,
                        gpu_devices=selected_settings.gpu_devices,
                        gpu_tile_size=selected_settings.gpu_tile_size,
                        max_gpu_support_bytes=selected_settings.max_gpu_support_bytes,
                    )
                operational_metrics["loo_seconds"] += max(0.0, time.perf_counter() - loo_started)
                operational_metrics["loo_pair_count"] += len(c_batch)
                active_support_metrics = operational_metrics.get(
                    "loo_active_support_metrics", {}
                )
                if not isinstance(active_support_metrics, Mapping):
                    active_support_metrics = {}
                actual_gpu_pairs = int(
                    active_support_metrics.get("gpu_fp64_evaluations", 0) or 0
                )
                actual_d2h_bytes = int(active_support_metrics.get("gpu_d2h_bytes", 0) or 0)
                independent_gpu_pairs = len(c_batch) * pool.descriptors.shape[0] ** 2
                independent_d2h_bytes = independent_gpu_pairs * np.dtype(np.float64).itemsize
                operational_metrics["c_batch_actual_gpu_fp64_evaluations"] += actual_gpu_pairs
                operational_metrics["c_batch_actual_d2h_bytes"] += actual_d2h_bytes
                operational_metrics["c_batch_measured_scan_savings"] += max(
                    0, independent_gpu_pairs - actual_gpu_pairs
                )
                operational_metrics["c_batch_measured_d2h_savings_bytes"] += max(
                    0, independent_d2h_bytes - actual_d2h_bytes
                )
                operational_metrics["support_size_per_c"].append(
                    {
                        "k": int(candidate_k),
                        "c_values": list(c_batch),
                        "support_size": list(
                            operational_metrics.get("support_size_per_candidate", [])
                        ),
                        "support_max": list(
                            operational_metrics.get("support_max_per_candidate", [])
                        ),
                        "conservative_candidate_count": list(
                            operational_metrics.get(
                                "conservative_candidate_count_per_candidate", []
                            )
                        ),
                        "conservative_candidate_max": list(
                            operational_metrics.get(
                                "conservative_candidate_max_per_candidate", []
                            )
                        ),
                        "gpu_fp64_evaluations": int(
                            actual_gpu_pairs
                        ),
                        "gpu_launch_count": int(
                            active_support_metrics.get("gpu_launch_count", 0) or 0
                        ),
                        "gpu_d2h_transfer_count": int(
                            active_support_metrics.get("gpu_d2h_transfer_count", 0) or 0
                        ),
                        "gpu_d2h_bytes": int(
                            actual_d2h_bytes
                        ),
                        "gpu_cpu_verification_seconds": float(
                            active_support_metrics.get("gpu_cpu_verification_seconds", 0.0)
                            or 0.0
                        ),
                    }
                )
                sample_resource_leases()
                if isinstance(index, (IndexedCPUNeighbourIndex, TiledCPUNeighbourIndex)):
                    metrics = index.memory_metrics()
                    logger.info(
                        "Bandwidth calibration c-batch completed: k=%d, c_batch=%s, "
                        "actual_support_mean=%s, actual_support_max=%s, "
                        "radius_queries=%s, pilot_support_samples=%s, "
                        "cache_hit=%s, cache_miss=%s, cache_eviction=%s, "
                        "cache_admission_rejected=%s, cache_bytes=%s, "
                        "cache_peak_bytes=%s, cache_time_saved_seconds=%s",
                        candidate_k,
                        ",".join(f"{value:g}" for value in c_batch),
                        metrics.get("mean_support_count"),
                        metrics.get("max_support_count"),
                        metrics.get("radius_queries"),
                        metrics.get("pilot_support_samples"),
                        metrics.get("cache_hit"),
                        metrics.get("cache_miss"),
                        metrics.get("cache_eviction"),
                        metrics.get("cache_admission_rejected"),
                        metrics.get("cache_bytes"),
                        metrics.get("cache_peak_bytes"),
                        metrics.get("cache_time_saved_seconds"),
                    )
                for candidate_c, frozen, objective in zip(
                    c_batch,
                    frozen_batch,
                    objectives,
                    strict=True,
                ):
                    if not objective.valid or objective.objective is None:
                        record_attempt(
                            CalibrationAttempt(
                                candidate_k,
                                float(candidate_c),
                                "invalid",
                                reason=objective.reason or "invalid leave-one-out objective",
                                bandwidth_fingerprint=frozen.fingerprint,
                            )
                        )
                        continue
                    attempt = CalibrationAttempt(
                        candidate_k,
                        float(candidate_c),
                        "valid",
                        objective=objective.objective,
                        bandwidth_fingerprint=frozen.fingerprint,
                    )
                    record_attempt(attempt)
                    if winner is None or objective.objective < winner[0]:
                        winner = (objective.objective, frozen, objective.probabilities)
                append_pair_timings(start_position, pair_started)
                publish_grid(None, retire_checkpoint_name=loo_checkpoint_name)
                grid_completed_commits += 1
                retire_loo_checkpoint(loo_checkpoint_name)
            except CheckpointValidationError:
                raise
            except (FloatingPointError, OverflowError, ValueError) as exc:
                for batch_index, candidate_c in enumerate(c_batch):
                    record_attempt(
                        CalibrationAttempt(
                            candidate_k,
                            float(candidate_c),
                            "invalid",
                            reason=str(exc),
                            bandwidth_fingerprint=(
                                frozen_batch[batch_index].fingerprint
                                if batch_index < len(frozen_batch)
                                else None
                            ),
                        )
                    )
                append_pair_timings(start_position, pair_started)
                publish_grid(None, retire_checkpoint_name=loo_checkpoint_name)
                grid_completed_commits += 1
                retire_loo_checkpoint(loo_checkpoint_name)

    if completed_combinations != total_combinations:
        raise CheckpointValidationError(
            "calibration grid terminated before every ordered pair was durably completed"
        )
    if winner is None:
        elapsed, _ = _progress_timing(
            completed_combinations,
            total_combinations,
            calibration_started,
        )
        logger.error(
            "Bandwidth calibration failed: no valid pair, attempted=%d, valid=%d, "
            "invalid=%d, elapsed=%.3fs",
            completed_combinations,
            valid_count,
            invalid_count,
            elapsed,
        )
        raise BandwidthCalibrationError(
            "no valid entropy bandwidth calibration pair exists; "
            + "; ".join(
                f"(k={attempt.k}, c={attempt.c}): {attempt.reason or 'invalid'}"
                for attempt in attempts
            ),
            attempts=attempts,
        )
    objective_value, selected_bandwidths, loo_probabilities = winner
    elapsed, _ = _progress_timing(
        completed_combinations,
        total_combinations,
        calibration_started,
    )
    operational_metrics["wall_seconds"] = elapsed
    operational_metrics.update(
        {
            "pairs_reused": pairs_reused,
            "pairs_computed": pairs_computed,
            "resumed_elapsed_wall_seconds": resumed_elapsed_wall_seconds,
            "process_elapsed_wall_seconds": elapsed,
            "estimated_time_saved_seconds": float(sum(pair_wall_seconds[:pairs_reused])),
        }
    )
    if grid_store is not None:
        grid_metrics = grid_store.metrics()
        for metric_name, metric_value in grid_metrics.items():
            key = (
                metric_name
                if metric_name.startswith("grid_checkpoint_")
                else f"grid_checkpoint_{metric_name}"
            )
            if metric_name == "checkpoint_wall_seconds":
                key = "grid_checkpoint_wall_seconds"
            operational_metrics[key] = metric_value
        operational_metrics["grid_checkpoint_commits"] = grid_completed_commits
        operational_metrics["grid_checkpoint_pointer_promotions"] = int(
            grid_metrics.get("commits", 0) or 0
        )
        operational_metrics["grid_checkpoint_bytes"] = int(
            grid_metrics.get("physical_bytes_written", 0) or 0
        )
        operational_metrics["grid_checkpoint_retention_generations"] = 2
    operational_metrics["pair_throughput_per_second"] = (
        completed_combinations / elapsed if elapsed > 0.0 else 0.0
    )
    if isinstance(index, (IndexedCPUNeighbourIndex, TiledCPUNeighbourIndex)):
        index_metrics = index.memory_metrics()
        operational_metrics.update(
            {
                "support_query_seconds": float(
                    index_metrics.get("radius_retrieval_seconds", 0.0) or 0.0
                ),
                "support_queries": int(index_metrics.get("radius_queries", 0) or 0),
                "distance_evaluations": int(index_metrics.get("distance_evaluations", 0) or 0),
                "unique_distance_evaluations": int(
                    index_metrics.get("unique_distance_evaluations", 0) or 0
                ),
                "support_count_min": index_metrics.get("min_support_count"),
                "support_count_mean": index_metrics.get("mean_support_count"),
                "support_count_max": index_metrics.get("max_support_count"),
                "support_density_mean": (
                    float(index_metrics.get("mean_support_count")) / pool.descriptors.shape[0]
                    if index_metrics.get("mean_support_count") is not None
                    else None
                ),
                "process_peak_rss_bytes": index_metrics.get("process_peak_rss_bytes"),
                "checkpoint_bytes": int(index_metrics.get("checkpoint_serialization_bytes", 0) or 0)
                + int(operational_metrics.get("loo_checkpoint_bytes", 0) or 0),
                "checkpoint_seconds": float(
                    index_metrics.get("checkpoint_serialization_seconds", 0.0) or 0.0
                )
                + float(operational_metrics.get("loo_checkpoint_seconds", 0.0) or 0.0),
            }
        )
    logger.info(
        "Bandwidth calibration completed: selected k=%d, c=%g, objective=%.12g, "
        "attempted=%d/%d (100.0%%), valid=%d, invalid=%d, elapsed=%.3fs, "
        "pairs_reused=%d, pairs_computed=%d, resumed_elapsed_wall_seconds=%.3f, "
        "estimated_time_saved_seconds=%.3f, grid_checkpoint_commits=%s, "
        "wall_seconds=%.6f, pair_throughput=%.3f, support_queries=%s, "
        "distance_evaluations=%s, process_peak_rss_bytes=%s, "
        "resource_lease_peak_bytes=%d, scratch_peak_bytes=%d",
        selected_bandwidths.k,
        selected_bandwidths.c,
        objective_value,
        completed_combinations,
        total_combinations,
        valid_count,
        invalid_count,
        elapsed,
        pairs_reused,
        pairs_computed,
        resumed_elapsed_wall_seconds,
        operational_metrics.get("estimated_time_saved_seconds", 0.0),
        operational_metrics.get("grid_checkpoint_commits", 0),
        operational_metrics["wall_seconds"],
        operational_metrics["pair_throughput_per_second"],
        operational_metrics.get("support_queries"),
        operational_metrics.get("distance_evaluations"),
        operational_metrics.get("process_peak_rss_bytes"),
        operational_metrics["resource_lease_peak_bytes"],
        operational_metrics["scratch_peak_bytes"],
    )
    calibration_payload = {
        "schema": "entropy-calibration-v1",
        "pool": pool.fingerprint,
        "candidate_ids": list(pool.candidate_ids),
        "row_candidate_indices": pool.row_candidate_indices.tolist(),
        "probabilities": pool.probabilities.tolist(),
        "transform": pool.transform_fingerprint,
        "neighbour_backend": selected_settings.backend,
        "neighbour_backend_version": selected_bandwidths.backend_version,
        "neighbour_backend_fingerprint": selected_bandwidths.backend_fingerprint,
        "kernel_family": KERNEL_FAMILY,
        "kernel_version": KERNEL_VERSION,
        "normalization": "source-column-all-targets-including-self",
        "loo_exclusion": "source-row-only",
        "optimizer": ("manual" if selected_settings.mode == "manual" else CALIBRATION_OPTIMIZER_ID),
        "optimizer_version": (
            "manual-v1" if selected_settings.mode == "manual" else CALIBRATION_OPTIMIZER_VERSION
        ),
        "mode": selected_settings.mode,
        "k_domain": list(domain_k),
        "c_domain": list(domain_c),
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
        "selected": {
            "k": selected_bandwidths.k,
            "c": selected_bandwidths.c,
            "bandwidth": selected_bandwidths.fingerprint,
        },
        "termination_reason": (
            "manual_parameter_evaluated"
            if selected_settings.mode == "manual"
            else "complete_ordered_grid_evaluated"
        ),
    }
    return BandwidthCalibrationResult(
        mode=selected_settings.mode,
        selected=selected_bandwidths,
        objective=objective_value,
        attempts=tuple(attempts),
        k_domain=tuple(domain_k),
        c_domain=tuple(domain_c),
        optimizer_id=("manual" if selected_settings.mode == "manual" else CALIBRATION_OPTIMIZER_ID),
        optimizer_version=(
            "manual-v1" if selected_settings.mode == "manual" else CALIBRATION_OPTIMIZER_VERSION
        ),
        evaluation_count=len(attempts),
        termination_reason=calibration_payload["termination_reason"],
        pool_fingerprint=pool.fingerprint,
        calibration_fingerprint=sha256_canonical_json(calibration_payload),
        loo_probabilities=loo_probabilities,
        backend=selected_bandwidths.backend,
        backend_version=selected_bandwidths.backend_version,
        backend_fingerprint=selected_bandwidths.backend_fingerprint,
        operational_metrics=operational_metrics,
    )


calibrate_entropy_bandwidth = calibrate_bandwidth
automatic_bandwidth_calibration = calibrate_bandwidth


__all__ = [
    "BandwidthCalibrationCapacityError",
    "BandwidthCalibrationError",
    "automatic_bandwidth_calibration",
    "build_entropy_pool",
    "calculate_bandwidths",
    "calculate_frozen_bandwidths",
    "calibrate_bandwidth",
    "calibrate_entropy_bandwidth",
    "compute_bandwidths",
    "compute_frozen_bandwidths",
    "candidate_weights",
    "derive_candidate_probabilities",
    "freeze_bandwidths",
    "prepare_entropy_pool",
]
