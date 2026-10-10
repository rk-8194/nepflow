"""Exact entropy-pool validation, frozen bandwidths, and bounded calibration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.stages.selection.representations import LocalEnvironmentRepresentation

from .kernels import evaluate_leave_one_out_objective
from .models import (
    BANDWIDTH_SCHEMA_VERSION,
    CALIBRATION_OPTIMIZER_ID,
    CALIBRATION_OPTIMIZER_VERSION,
    KERNEL_FAMILY,
    KERNEL_VERSION,
    BandwidthCalibrationResult,
    CalibrationAttempt,
    EntropyBandwidthSettings,
    EntropyPool,
    FrozenBandwidths,
)
from .neighbours import ExactNeighbourResult, compute_exact_neighbours


class BandwidthCalibrationError(ValueError):
    """Raised when no scientifically valid bandwidth calibration exists."""

    def __init__(self, message: str, *, attempts: Sequence[CalibrationAttempt] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


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
    neighbours = compute_exact_neighbours(
        pool.descriptors,
        k,
        chunk_size=chunk_size,
        representation_fingerprint=pool.representation_fingerprint,
        row_ids=_pool_row_ids(pool),
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
        )
    return EntropyBandwidthSettings(
        mode=selected_mode,
        k_candidates=tuple(k_candidates) if k_candidates is not None else (1, 2, 4, 8),
        c_candidates=tuple(c_candidates) if c_candidates is not None else (1.5, 2.0, 4.0, 8.0),
        backend=backend,
        metric=metric,
        chunk_size=chunk_size,
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
    backend: str = "exact_cpu",
    metric: str = "euclidean",
    chunk_size: int = 1024,
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
    )
    if pool.descriptors.shape[0] < 2:
        raise BandwidthCalibrationError("leave-one-out calibration requires at least two rows")

    if selected_settings.mode == "manual":
        if selected_settings.k is None or selected_settings.c is None:
            raise ValueError("manual bandwidth settings require explicit k and c")
        domain_k = (int(selected_settings.k),)
        domain_c = (float(selected_settings.c),)
    else:
        domain_k = selected_settings.k_candidates
        domain_c = selected_settings.c_candidates

    attempts: list[CalibrationAttempt] = []
    radii_by_k: dict[int, ExactNeighbourResult | str] = {}
    winner: tuple[float, FrozenBandwidths, np.ndarray] | None = None
    for candidate_k in domain_k:
        try:
            neighbours = radii_by_k.get(candidate_k)
            if neighbours is None:
                neighbours = compute_exact_neighbours(
                    pool.descriptors,
                    candidate_k,
                    chunk_size=selected_settings.chunk_size,
                    representation_fingerprint=pool.representation_fingerprint,
                    row_ids=_pool_row_ids(pool),
                )
                radii_by_k[candidate_k] = neighbours
            if isinstance(neighbours, str):
                raise ValueError(neighbours)
        except (ValueError, TypeError) as exc:
            reason = str(exc)
            radii_by_k[candidate_k] = reason
            for candidate_c in domain_c:
                attempts.append(
                    CalibrationAttempt(candidate_k, float(candidate_c), "invalid", reason=reason)
                )
            continue

        for candidate_c in domain_c:
            try:
                frozen = _freeze_from_neighbours(pool, neighbours, candidate_c)
                objective = evaluate_leave_one_out_objective(
                    pool.descriptors,
                    pool.probabilities,
                    frozen.bandwidths,
                    chunk_size=selected_settings.chunk_size,
                )
                if not objective.valid or objective.objective is None:
                    attempts.append(
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
                attempts.append(attempt)
                if winner is None or objective.objective < winner[0]:
                    winner = (objective.objective, frozen, objective.probabilities)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                attempts.append(
                    CalibrationAttempt(
                        candidate_k,
                        float(candidate_c),
                        "invalid",
                        reason=str(exc),
                    )
                )

        if selected_settings.mode == "manual":
            break

    if winner is None:
        raise BandwidthCalibrationError(
            "no valid entropy bandwidth calibration pair exists; "
            + "; ".join(
                f"(k={attempt.k}, c={attempt.c}): {attempt.reason or 'invalid'}"
                for attempt in attempts
            ),
            attempts=attempts,
        )
    objective_value, selected_bandwidths, loo_probabilities = winner
    calibration_payload = {
        "schema": "entropy-calibration-v1",
        "pool": pool.fingerprint,
        "candidate_ids": list(pool.candidate_ids),
        "row_candidate_indices": pool.row_candidate_indices.tolist(),
        "probabilities": pool.probabilities.tolist(),
        "transform": pool.transform_fingerprint,
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
    )


calibrate_entropy_bandwidth = calibrate_bandwidth
automatic_bandwidth_calibration = calibrate_bandwidth


__all__ = [
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
