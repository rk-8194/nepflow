"""Exact entropy-pool validation, frozen bandwidths, and bounded calibration."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)
from nepflow.stages.selection.representations import LocalEnvironmentRepresentation

from .kernels import (
    evaluate_leave_one_out_objective,
    evaluate_leave_one_out_objectives,
)
from .models import (
    BANDWIDTH_SCHEMA_VERSION,
    CALIBRATION_OPTIMIZER_ID,
    CALIBRATION_OPTIMIZER_VERSION,
    DEFAULT_NEIGHBOUR_BACKEND_ID,
    KERNEL_FAMILY,
    KERNEL_VERSION,
    NEIGHBOUR_METRIC,
    BandwidthCalibrationResult,
    CalibrationAttempt,
    EntropyBandwidthSettings,
    EntropyPool,
    FrozenBandwidths,
)
from .neighbours import (
    ExactNeighbourResult,
    RadiusQueryCapacityError,
    _estimate_radius_workspace_bytes,
    build_neighbour_index,
    compute_neighbours,
)

logger = logging.getLogger(__name__)


class BandwidthCalibrationError(ValueError):
    """Raised when no scientifically valid bandwidth calibration exists."""

    def __init__(self, message: str, *, attempts: Sequence[CalibrationAttempt] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


BandwidthCalibrationCapacityError = RadiusQueryCapacityError


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
    index = build_neighbour_index(
        pool.descriptors,
        backend=backend,
        max_index_bytes=(
            remaining if max_index_bytes is None else min(max_index_bytes, remaining)
        ),
        max_radius_query_bytes=(
            max(1, remaining // 2)
            if max_radius_query_bytes is None
            else min(max_radius_query_bytes, max(1, remaining))
        ),
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
    max_calibration_work_bytes: int | None,
    calibration_batch_size: int | None,
) -> EntropyBandwidthSettings:
    if settings is not None:
        if any(value is not None for value in (mode, k, c, k_candidates, c_candidates)):
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
            max_calibration_work_bytes=max_calibration_work_bytes,
            calibration_batch_size=calibration_batch_size,
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
        max_calibration_work_bytes=max_calibration_work_bytes,
        calibration_batch_size=calibration_batch_size,
    )


def _attempt_payload(attempt: CalibrationAttempt) -> dict[str, Any]:
    return {
        "k": attempt.k,
        "c": attempt.c,
        "status": attempt.status,
        "objective": attempt.objective,
        "reason": attempt.reason,
        "bandwidth_fingerprint": attempt.bandwidth_fingerprint,
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
    max_calibration_work_bytes: int | None = None,
    calibration_batch_size: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
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
        max_calibration_work_bytes=max_calibration_work_bytes,
        calibration_batch_size=calibration_batch_size,
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
        max_index_bytes=selected_settings.max_index_bytes,
        max_radius_query_bytes=selected_settings.radius_query_bytes,
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

    attempts: list[CalibrationAttempt] = []
    radii_by_k: dict[int, ExactNeighbourResult | str] = {}
    winner: tuple[float, FrozenBandwidths, np.ndarray] | None = None
    completed_combinations = 0
    valid_count = 0
    invalid_count = 0

    def record_attempt(attempt: CalibrationAttempt) -> None:
        nonlocal completed_combinations, valid_count, invalid_count
        attempts.append(attempt)
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
                "estimated remaining=%.3fs",
                *progress_values[:2],
                attempt.objective,
                *progress_values[2:],
            )
        else:
            logger.info(
                "Bandwidth calibration pair completed: k=%d, c=%g, invalid=%s, "
                "%d/%d combinations (%.1f%%), valid=%d, invalid=%d, elapsed=%.3fs, "
                "estimated remaining=%.3fs",
                *progress_values[:2],
                attempt.reason or "unspecified invalidity",
                *progress_values[2:],
            )

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

        def report(completed: int, total_sources: int) -> None:
            nonlocal next_percent, last_completed
            if completed <= last_completed:
                return
            last_completed = completed
            if total_sources < 20:
                should_report = True
            else:
                should_report = completed * 100 >= next_percent * total_sources
            if not should_report:
                return
            elapsed, remaining = _progress_timing(completed, total_sources, pair_started)
            percentage = 100.0 * completed / total_sources
            metrics: Mapping[str, int | float | None] = (
                index.memory_metrics() if index is not None else {}
            )

            def metric(name: str, default: int | float = 0) -> int | float:
                value = metrics.get(name)
                return default if value is None else value

            logger.info(
                "Bandwidth calibration pair source progress: k=%d, %s, source=%d/%d "
                "(%.1f%%), current_support=%d, min_support=%s, mean_support=%s, "
                "max_support=%d, "
                "radius_queries=%d, unique_distance_evaluations=%d, "
                "atomic_distance_evaluations=%d, "
                "support_cache_bytes=0, "
                "support_cache_peak_bytes=0, grouping_bytes=%d, grouping_build_seconds=%.3f, "
                "tree_build_seconds=%.3f, index_bytes=%d, query_workspace_peak_bytes=%d, "
                "process_peak_rss_bytes=%s, cgroup_memory_current_bytes=%s, "
                "elapsed=%.3fs, estimated remaining=%.3fs",
                candidate_k,
                c_label,
                completed,
                total_sources,
                percentage,
                int(metric("current_support_count")),
                str(metrics.get("min_support_count", "unavailable")),
                str(metrics.get("mean_support_count", "unavailable")),
                int(metric("max_support_count")),
                int(metric("radius_queries")),
                int(metric("unique_distance_evaluations")),
                int(metric("atomic_distance_evaluations")),
                int(metric("grouping_bytes")),
                float(metric("grouping_build_seconds")),
                float(metric("tree_build_seconds")),
                int(metric("index_bytes")),
                int(metric("query_workspace_peak_bytes")),
                str(metrics.get("process_peak_rss_bytes", "unavailable")),
                str(metrics.get("cgroup_memory_current_bytes", "unavailable")),
                elapsed,
                remaining,
            )
            if total_sources >= 20:
                while next_percent <= 100 and completed * 100 >= next_percent * total_sources:
                    next_percent += 5

        return report

    def c_batches() -> tuple[tuple[float, ...], ...]:
        ordered = tuple(float(value) for value in domain_c)
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
        logger.info(
            "Bandwidth calibration c batching: batch_size=%d, c_count=%d, c_batches=%d, "
            "estimated_batch_workspace_bytes=%d, max_calibration_work_bytes=%d",
            width,
            len(ordered),
            (len(ordered) + width - 1) // width,
            8192 + width * per_c_bytes + source_workspace_bound,
            selected_settings.operational_work_bytes,
        )
        return tuple(ordered[start : start + width] for start in range(0, len(ordered), width))

    batches = c_batches()

    for candidate_k in domain_k:
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
                )
                radii_by_k[candidate_k] = neighbours
            if isinstance(neighbours, str):
                raise ValueError(neighbours)
        except (ValueError, TypeError) as exc:
            reason = str(exc)
            radii_by_k[candidate_k] = reason
            for candidate_c in domain_c:
                record_attempt(
                    CalibrationAttempt(candidate_k, float(candidate_c), "invalid", reason=reason)
                )
            continue

        for c_batch in batches:
            try:
                frozen_batch = tuple(
                    _freeze_from_neighbours(pool, neighbours, candidate_c)
                    for candidate_c in c_batch
                )
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
            except (FloatingPointError, OverflowError, ValueError) as exc:
                for candidate_c in c_batch:
                    record_attempt(
                        CalibrationAttempt(
                            candidate_k,
                            float(candidate_c),
                            "invalid",
                            reason=str(exc),
                        )
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
    logger.info(
        "Bandwidth calibration completed: selected k=%d, c=%g, objective=%.12g, "
        "attempted=%d/%d (100.0%%), valid=%d, invalid=%d, elapsed=%.3fs",
        selected_bandwidths.k,
        selected_bandwidths.c,
        objective_value,
        completed_combinations,
        total_combinations,
        valid_count,
        invalid_count,
        elapsed,
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
