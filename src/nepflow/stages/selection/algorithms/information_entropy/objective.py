"""Exact sparse information-theoretic objective and marginal gains."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

from .models import (
    ENTROPY_OBJECTIVE_SCHEMA_VERSION,
    SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION,
    SPARSE_NUMERICAL_TOLERANCE,
    EntropyObjectiveDiagnostics,
    EntropyObjectiveState,
    EntropyPool,
    SparseCandidateContributionRow,
    SparseCandidateContributions,
)

ObjectiveCandidate = int | str


def _validate_beta(beta: Any) -> float:
    if isinstance(beta, bool):
        raise ValueError("beta must be finite and strictly positive")
    try:
        value = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be finite and strictly positive") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return value


def _validate_budget(value: Any, candidate_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("objective budget must be an integer")
    budget = int(value)
    if budget < 0 or budget > candidate_count:
        raise ValueError("objective budget must satisfy 0 <= budget <= candidate count")
    return budget


def _validate_identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _validate_objective_inputs(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
) -> tuple[int, tuple[str, ...], str]:
    if not isinstance(pool, EntropyPool):
        raise TypeError("objective requires an EntropyPool")
    if not isinstance(contributions, SparseCandidateContributions):
        raise TypeError("objective requires SparseCandidateContributions")
    pool_fingerprint = _validate_identity(pool.fingerprint, name="pool fingerprint")
    contribution_pool_fingerprint = _validate_identity(
        contributions.pool_fingerprint,
        name="contribution pool fingerprint",
    )
    if contribution_pool_fingerprint != pool_fingerprint:
        raise ValueError("candidate contributions belong to a different entropy pool")
    _validate_identity(contributions.graph_fingerprint, name="graph fingerprint")
    contribution_fingerprint = _validate_identity(
        contributions.fingerprint,
        name="contribution fingerprint",
    )
    if contributions.sparse_schema_version != SPARSE_CANDIDATE_CONTRIBUTION_SCHEMA_VERSION:
        raise ValueError("candidate contributions use an incompatible schema version")
    if not math.isclose(
        contributions.numerical_tolerance,
        SPARSE_NUMERICAL_TOLERANCE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("candidate contributions use an incompatible numerical tolerance")

    candidate_ids = tuple(pool.candidate_ids)
    contribution_ids = tuple(contributions.candidate_ids)
    if not candidate_ids or any(
        not isinstance(value, str) or not value.strip() for value in candidate_ids
    ):
        raise ValueError("pool candidate IDs must be non-empty and non-blank")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("pool candidate IDs must be unique")
    if contribution_ids != candidate_ids:
        raise ValueError("candidate contribution IDs do not match the ordered pool IDs")
    if len(set(contribution_ids)) != len(contribution_ids):
        raise ValueError("candidate contribution IDs must be unique")

    probabilities = np.asarray(pool.probabilities)
    row_candidates = np.asarray(pool.row_candidate_indices)
    target_count = len(pool.rows)
    if probabilities.dtype != np.dtype(np.float64) or probabilities.ndim != 1:
        raise ValueError("pool probabilities must be a one-dimensional float64 vector")
    if probabilities.shape[0] != target_count:
        raise ValueError("pool probabilities and rows must have the same length")
    if row_candidates.dtype != np.dtype(np.int64) or row_candidates.shape != (target_count,):
        raise ValueError("pool row ownership must be an int64 vector aligned with rows")
    if np.any(row_candidates < 0) or np.any(row_candidates >= len(candidate_ids)):
        raise ValueError("pool row ownership contains an unknown candidate")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("pool probabilities must be finite and strictly positive")
    if not math.isclose(
        float(np.sum(probabilities, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=SPARSE_NUMERICAL_TOLERANCE,
    ):
        raise ValueError("pool probabilities must sum to one")

    candidate_count = len(candidate_ids)
    row_counts = np.bincount(row_candidates, minlength=candidate_count)
    if np.any(row_counts <= 0):
        missing = candidate_ids[int(np.flatnonzero(row_counts <= 0)[0])]
        raise ValueError(f"candidate {missing!r} owns no pool rows")
    expected_probabilities = np.asarray(
        [1.0 / (candidate_count * int(row_counts[index])) for index in row_candidates],
        dtype=np.float64,
    )
    if not np.allclose(
        probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=SPARSE_NUMERICAL_TOLERANCE,
    ):
        raise ValueError("pool probabilities do not equal the canonical candidate-row masses")
    candidate_masses = np.bincount(
        row_candidates,
        weights=probabilities,
        minlength=candidate_count,
    )
    if not np.allclose(
        candidate_masses,
        np.full(candidate_count, 1.0 / candidate_count, dtype=np.float64),
        rtol=0.0,
        atol=SPARSE_NUMERICAL_TOLERANCE,
    ):
        raise ValueError("each candidate must carry target probability mass 1/M")

    if contributions.row_count != target_count:
        raise ValueError("candidate contribution row_count does not match the entropy pool")
    for candidate_index, candidate_id in enumerate(candidate_ids):
        row = contributions.candidate_row(candidate_index)
        _validate_contribution_row(row, target_count, candidate_id)
    return target_count, candidate_ids, contribution_fingerprint


def _validate_contribution_row(
    row: SparseCandidateContributionRow,
    target_count: int,
    candidate_id: str,
) -> None:
    if row.candidate_id != candidate_id:
        raise ValueError("candidate contribution row identity does not match candidate_ids")
    targets = np.asarray(row.target_indices)
    values = np.asarray(row.values)
    if targets.dtype != np.dtype(np.int64) or values.dtype != np.dtype(np.float64):
        raise ValueError("candidate contribution support must use int64/float64 arrays")
    if targets.ndim != 1 or values.ndim != 1 or targets.shape != values.shape:
        raise ValueError("candidate contribution support and values must be aligned vectors")
    if targets.size == 0:
        raise ValueError(f"candidate {candidate_id!r} has empty support")
    if np.any(targets < 0) or np.any(targets >= target_count):
        raise ValueError(f"candidate {candidate_id!r} has an out-of-range target index")
    if targets.size > 1 and np.any(np.diff(targets) <= 0):
        raise ValueError(f"candidate {candidate_id!r} target indices are not strictly increasing")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"candidate {candidate_id!r} values must be finite and positive")
    if not math.isclose(
        float(np.sum(values, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=SPARSE_NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"candidate {candidate_id!r} contribution must sum to one")


def _state_fingerprint(
    *,
    pool_fingerprint: str,
    graph_fingerprint: str,
    contributions_fingerprint: str,
    selected_indices: tuple[int, ...],
    beta: float,
    budget: int,
    objective: float,
    anchor_objective: float,
) -> str:
    payload = {
        "schema": ENTROPY_OBJECTIVE_SCHEMA_VERSION,
        "pool": pool_fingerprint,
        "graph": graph_fingerprint,
        "contributions": contributions_fingerprint,
        "selected_indices": list(selected_indices),
        "beta": beta,
        "budget": budget,
        "objective": objective,
        "anchor_objective": anchor_objective,
        "numerical_tolerance": SPARSE_NUMERICAL_TOLERANCE,
    }
    return sha256_canonical_json(payload)


def _objective_from_arrays(probabilities: np.ndarray, support: np.ndarray) -> float:
    if probabilities.shape != support.shape:
        raise ValueError("objective probabilities and support must be aligned")
    if (
        probabilities.dtype != np.dtype(np.float64)
        or support.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(support))
        or np.any(probabilities <= 0.0)
        or np.any(support <= 0.0)
    ):
        raise ValueError("objective probabilities and support must be finite and positive float64")
    with np.errstate(divide="ignore", invalid="raise", over="raise"):
        try:
            terms = probabilities * np.log(support)
            objective = float(np.sum(terms, dtype=np.float64))
        except FloatingPointError as exc:
            raise ValueError("information objective evaluation became non-finite") from exc
    if not math.isfinite(objective):
        raise ValueError("information objective evaluation became non-finite")
    return objective


def _resolve_candidate(candidate_ids: tuple[str, ...], candidate: ObjectiveCandidate) -> int:
    if isinstance(candidate, bool):
        raise TypeError("candidate must be a candidate ID or integer index")
    if isinstance(candidate, str):
        try:
            return candidate_ids.index(candidate)
        except ValueError as exc:
            raise KeyError(f"unknown candidate ID: {candidate!r}") from exc
    if not isinstance(candidate, (int, np.integer)):
        raise TypeError("candidate must be a candidate ID or integer index")
    index = int(candidate)
    if index < 0 or index >= len(candidate_ids):
        raise IndexError("candidate index is outside the objective pool")
    return index


def _select_candidate_argument(
    candidate: ObjectiveCandidate | None,
    candidate_id: ObjectiveCandidate | None,
) -> ObjectiveCandidate:
    if candidate is not None and candidate_id is not None:
        raise ValueError("supply only one of candidate or candidate_id")
    selected = candidate if candidate is not None else candidate_id
    if selected is None:
        raise ValueError("an objective candidate is required")
    return selected


def _validate_state_compatibility(
    state: EntropyObjectiveState,
    contributions: SparseCandidateContributions,
) -> None:
    if not isinstance(state, EntropyObjectiveState):
        raise TypeError("objective operation requires EntropyObjectiveState")
    if not isinstance(contributions, SparseCandidateContributions):
        raise TypeError("objective operation requires SparseCandidateContributions")
    if state.contributions_fingerprint != contributions.fingerprint:
        raise ValueError("objective state belongs to different candidate contributions")
    if state.pool_fingerprint != contributions.pool_fingerprint:
        raise ValueError("objective state and contributions have mismatched pool identities")
    if state.graph_fingerprint != contributions.graph_fingerprint:
        raise ValueError("objective state belongs to different kernel graph contributions")
    if state.candidate_ids != tuple(contributions.candidate_ids):
        raise ValueError("objective state and contributions have different candidate ordering")
    if state.n_targets != contributions.row_count:
        raise ValueError("objective state and contributions have different target counts")
    if state.probabilities.shape != (contributions.row_count,):
        raise ValueError("objective state probabilities have the wrong target count")


def _support_gain(
    state: EntropyObjectiveState,
    row: SparseCandidateContributionRow,
) -> float:
    _validate_contribution_row(row, state.n_targets, row.candidate_id)
    targets = row.target_indices
    current = state.s[targets]
    if not np.all(np.isfinite(current)) or np.any(current <= 0.0):
        raise ValueError("objective state support must be finite and strictly positive")
    with np.errstate(divide="ignore", invalid="raise", over="raise"):
        try:
            ratio = row.values / current
            if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0.0):
                raise ValueError("candidate marginal ratio underflowed or became invalid")
            terms = state.probabilities[targets] * np.log1p(ratio)
            gain = float(np.sum(terms, dtype=np.float64))
        except FloatingPointError as exc:
            raise ValueError("candidate marginal gain became non-finite") from exc
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("candidate marginal gain became invalid")
    if gain == 0.0:
        raise ValueError("candidate marginal gain underflowed to zero")
    return gain


def initialize_entropy_objective(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta: float,
    anchors: Iterable[str] | None = (),
    budget: int | None = None,
    *,
    final_budget: int | None = None,
    target_count: int | None = None,
    K: int | None = None,
    anchor_ids: Iterable[str] | None = None,
) -> EntropyObjectiveState:
    """Create an anchor-initialised objective state for a fixed final budget."""

    target_size, candidate_ids, contribution_fingerprint = _validate_objective_inputs(
        pool, contributions
    )
    budget_aliases = [
        value for value in (budget, final_budget, target_count, K) if value is not None
    ]
    if len(budget_aliases) > 1:
        raise ValueError("supply only one objective final budget")
    if budget_aliases:
        budget = budget_aliases[0]
    if budget is None:
        raise ValueError("objective initialization requires a final budget")
    budget_value = _validate_budget(budget, len(candidate_ids))
    beta_value = _validate_beta(beta)
    if anchor_ids is not None:
        if anchors not in ((), None):
            raise ValueError("supply only one of anchors or anchor_ids")
        anchors = anchor_ids
    if anchors is None:
        anchor_ids = ()
    elif isinstance(anchors, str):
        anchor_ids = (anchors,)
    else:
        try:
            anchor_ids = tuple(anchors)
        except TypeError as exc:
            raise ValueError("anchors must be an iterable of candidate IDs") from exc
    if any(not isinstance(value, str) or not value.strip() for value in anchor_ids):
        raise ValueError("anchors must contain non-blank candidate IDs")
    anchor_indices: list[int] = []
    for candidate_id in anchor_ids:
        index = _resolve_candidate(candidate_ids, candidate_id)
        if index in anchor_indices:
            raise ValueError(f"duplicate anchor candidate ID: {candidate_id!r}")
        anchor_indices.append(index)
    if len(anchor_indices) > budget_value:
        raise ValueError("anchor count exceeds objective budget")
    anchor_indices.sort()

    probabilities = np.array(pool.probabilities, dtype=np.float64, copy=True)
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        try:
            baseline = beta_value * probabilities
        except FloatingPointError as exc:
            raise ValueError("objective baseline overflowed") from exc
    if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0.0):
        raise ValueError("beta times target probabilities overflowed or underflowed")
    support = np.array(baseline, dtype=np.float64, copy=True)
    for candidate_index in anchor_indices:
        row = contributions.candidate_row(candidate_index)
        _validate_contribution_row(row, target_size, candidate_ids[candidate_index])
        with np.errstate(over="raise", invalid="raise"):
            try:
                updated = support[row.target_indices] + row.values
            except FloatingPointError as exc:
                raise ValueError("anchor contribution overflowed objective support") from exc
        if not np.all(np.isfinite(updated)) or np.any(updated <= 0.0):
            raise ValueError("anchor contribution produced invalid objective support")
        support[row.target_indices] = updated
    expected_mass = beta_value + len(anchor_indices)
    actual_mass = float(np.sum(support, dtype=np.float64))
    if not math.isclose(
        actual_mass,
        expected_mass,
        rel_tol=0.0,
        abs_tol=SPARSE_NUMERICAL_TOLERANCE,
    ):
        raise ValueError("anchor objective support mass does not match beta plus anchor count")
    objective = _objective_from_arrays(probabilities, support)
    fingerprint = _state_fingerprint(
        pool_fingerprint=pool.fingerprint,
        graph_fingerprint=contributions.graph_fingerprint,
        contributions_fingerprint=contribution_fingerprint,
        selected_indices=tuple(anchor_indices),
        beta=beta_value,
        budget=budget_value,
        objective=objective,
        anchor_objective=objective,
    )
    return EntropyObjectiveState(
        probabilities=probabilities,
        baseline=baseline,
        s=support,
        candidate_ids=candidate_ids,
        selected_indices=tuple(anchor_indices),
        selected_candidate_ids=tuple(candidate_ids[index] for index in anchor_indices),
        budget=budget_value,
        beta=beta_value,
        objective=objective,
        anchor_objective=objective,
        pool_fingerprint=pool.fingerprint,
        graph_fingerprint=contributions.graph_fingerprint,
        contributions_fingerprint=contribution_fingerprint,
        fingerprint=fingerprint,
    )


def objective_value(state: EntropyObjectiveState) -> float:
    """Return the cached ``F(A)`` value."""

    if not isinstance(state, EntropyObjectiveState):
        raise TypeError("objective_value requires EntropyObjectiveState")
    if not math.isfinite(state.objective):
        raise ValueError("objective value is non-finite")
    return float(state.objective)


def recompute_objective(state: EntropyObjectiveState) -> float:
    """Recompute ``F(A)`` as a bounded reference check over the target vector."""

    if not isinstance(state, EntropyObjectiveState):
        raise TypeError("recompute_objective requires EntropyObjectiveState")
    return _objective_from_arrays(state.probabilities, state.s)


def marginal_gain(
    state: EntropyObjectiveState,
    contributions: SparseCandidateContributions,
    candidate: ObjectiveCandidate | None = None,
    *,
    candidate_id: ObjectiveCandidate | None = None,
) -> float:
    """Return exact ``Delta F(C | A)`` using only candidate support."""

    _validate_state_compatibility(state, contributions)
    selected_candidate = _select_candidate_argument(candidate, candidate_id)
    candidate_index = _resolve_candidate(state.candidate_ids, selected_candidate)
    if candidate_index in state.selected_indices:
        raise ValueError(f"candidate {state.candidate_ids[candidate_index]!r} is already selected")
    row = contributions.candidate_row(candidate_index)
    return _support_gain(state, row)


def apply_candidate(
    state: EntropyObjectiveState,
    contributions: SparseCandidateContributions,
    candidate: ObjectiveCandidate | None = None,
    *,
    candidate_id: ObjectiveCandidate | None = None,
) -> EntropyObjectiveState:
    """Apply one unselected candidate to the owned objective state in-place."""

    _validate_state_compatibility(state, contributions)
    selected_candidate = _select_candidate_argument(candidate, candidate_id)
    candidate_index = _resolve_candidate(state.candidate_ids, selected_candidate)
    if candidate_index in state.selected_indices:
        raise ValueError(f"candidate {state.candidate_ids[candidate_index]!r} is already selected")
    if state.selected_count >= state.budget:
        raise ValueError("objective budget is already complete")
    row = contributions.candidate_row(candidate_index)
    gain = _support_gain(state, row)
    with np.errstate(over="raise", invalid="raise"):
        try:
            updated = state.s[row.target_indices] + row.values
            new_objective = state.objective + gain
        except FloatingPointError as exc:
            raise ValueError("candidate update overflowed objective state") from exc
    if not np.all(np.isfinite(updated)) or np.any(updated <= 0.0):
        raise ValueError("candidate update produced invalid objective support")
    if not math.isfinite(new_objective):
        raise ValueError("candidate update produced a non-finite objective")
    new_indices = state.selected_indices + (candidate_index,)
    new_ids = state.selected_candidate_ids + (state.candidate_ids[candidate_index],)
    state.s[row.target_indices] = updated
    state.selected_indices = new_indices
    state.selected_candidate_ids = new_ids
    state.objective = new_objective
    state.fingerprint = _state_fingerprint(
        pool_fingerprint=state.pool_fingerprint,
        graph_fingerprint=state.graph_fingerprint,
        contributions_fingerprint=state.contributions_fingerprint,
        selected_indices=new_indices,
        beta=state.beta,
        budget=state.budget,
        objective=new_objective,
        anchor_objective=state.anchor_objective,
    )
    return state


def objective_diagnostics(state: EntropyObjectiveState) -> EntropyObjectiveDiagnostics:
    """Return current-prefix cross-entropy, Shannon entropy and forward KL."""

    if not isinstance(state, EntropyObjectiveState):
        raise TypeError("objective_diagnostics requires EntropyObjectiveState")
    objective = recompute_objective(state)
    if not math.isclose(
        objective,
        state.objective,
        rel_tol=0.0,
        abs_tol=state.numerical_tolerance,
    ):
        raise ValueError("cached objective is inconsistent with objective support")
    denominator = state.beta + state.selected_count
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("objective normalization denominator is invalid")
    total_mass = float(np.sum(state.s, dtype=np.float64))
    if not math.isclose(
        total_mass,
        denominator,
        rel_tol=0.0,
        abs_tol=state.numerical_tolerance,
    ):
        raise ValueError("objective support mass is inconsistent with selected count")
    with np.errstate(divide="ignore", invalid="raise", over="raise"):
        try:
            normalized = state.s / denominator
            if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
                raise ValueError("normalized objective support is invalid")
            cross_entropy = float(
                -np.sum(state.probabilities * np.log(normalized), dtype=np.float64)
            )
            shannon_entropy = float(
                -np.sum(state.probabilities * np.log(state.probabilities), dtype=np.float64)
            )
        except FloatingPointError as exc:
            raise ValueError("objective entropy diagnostics became non-finite") from exc
    if not math.isfinite(cross_entropy) or not math.isfinite(shannon_entropy):
        raise ValueError("objective entropy diagnostics became non-finite")
    kl_divergence = cross_entropy - shannon_entropy
    if not math.isfinite(kl_divergence):
        raise ValueError("objective KL divergence became non-finite")
    if kl_divergence < -state.numerical_tolerance:
        raise ValueError("forward KL divergence is materially negative")
    if kl_divergence < 0.0:
        kl_divergence = 0.0
    return EntropyObjectiveDiagnostics(
        selected_count=state.selected_count,
        budget=state.budget,
        denominator=denominator,
        total_mass=total_mass,
        objective=objective,
        anchor_normalized_objective=objective - state.anchor_objective,
        cross_entropy=cross_entropy,
        shannon_entropy=shannon_entropy,
        kl_divergence=kl_divergence,
        beta=state.beta,
        is_final=state.is_final,
        state_fingerprint=state.fingerprint,
    )


def anchor_normalized_objective(state: EntropyObjectiveState) -> float:
    """Return ``F(A) - F(A_0)`` for the current state."""

    if not isinstance(state, EntropyObjectiveState):
        raise TypeError("anchor_normalized_objective requires EntropyObjectiveState")
    value = state.anchor_normalized_objective
    if not math.isfinite(value):
        raise ValueError("anchor-normalized objective is non-finite")
    return float(value)


def cross_entropy(state: EntropyObjectiveState) -> float:
    return objective_diagnostics(state).cross_entropy


def shannon_entropy(state: EntropyObjectiveState) -> float:
    return objective_diagnostics(state).shannon_entropy


def forward_kl_divergence(state: EntropyObjectiveState) -> float:
    return objective_diagnostics(state).kl_divergence


initialize_objective_state = initialize_entropy_objective
create_entropy_objective_state = initialize_entropy_objective
initialize_entropy_objective_state = initialize_entropy_objective
initialise_entropy_objective = initialize_entropy_objective
initialise_objective_state = initialize_entropy_objective
calculate_marginal_gain = marginal_gain
compute_marginal_gain = marginal_gain
candidate_marginal_gain = marginal_gain
update_objective_state = apply_candidate
apply_objective_update = apply_candidate
apply_selected_candidate = apply_candidate
evaluate_objective = objective_value
evaluate_objective_diagnostics = objective_diagnostics
calculate_objective = objective_value
kl_divergence = forward_kl_divergence


__all__ = [
    "ObjectiveCandidate",
    "anchor_normalized_objective",
    "apply_candidate",
    "apply_objective_update",
    "apply_selected_candidate",
    "calculate_objective",
    "calculate_marginal_gain",
    "candidate_marginal_gain",
    "compute_marginal_gain",
    "create_entropy_objective_state",
    "cross_entropy",
    "evaluate_objective",
    "evaluate_objective_diagnostics",
    "forward_kl_divergence",
    "initialize_entropy_objective",
    "initialize_entropy_objective_state",
    "initialize_objective_state",
    "initialise_entropy_objective",
    "initialise_objective_state",
    "kl_divergence",
    "marginal_gain",
    "objective_diagnostics",
    "objective_value",
    "recompute_objective",
    "shannon_entropy",
    "update_objective_state",
]
