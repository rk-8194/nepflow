"""Deterministic complete-candidate information-entropy selection."""

from __future__ import annotations

import heapq
import logging
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from nepflow.resources.budget import ResourceBudgetService, build_resource_budget
from nepflow.stages.selection.representations import LocalEnvironmentRepresentation

from ..base import SelectionAlgorithmRequest, SelectionAlgorithmResult
from .bandwidth import build_entropy_pool, calibrate_bandwidth
from .kernels import build_streamed_candidate_contributions
from .models import (
    EntropyBandwidthSettings,
    EntropyObjectiveState,
    EntropyPool,
    EntropySelectionHistory,
    EntropySelectionPerformance,
    EntropySelectionResult,
    EntropySelectionStep,
    SparseCandidateContributions,
)
from .objective import (
    apply_candidate,
    initialize_entropy_objective,
    marginal_gain,
    objective_diagnostics,
)

LOGGER = logging.getLogger(__name__)
GREEDY_SELECTION_VERSION = "information-entropy-greedy-v1"


class EntropySelectionError(ValueError):
    """Base error for invalid or uncertifiable direct entropy selections."""


class EntropySelectionCertificationError(EntropySelectionError):
    """Raised when finite-precision bounds cannot certify a deterministic winner."""


SelectionCertificationError = EntropySelectionCertificationError


@dataclass(slots=True)
class _SelectionCounters:
    gain_evaluations: int = 0
    initial_gain_evaluations: int = 0
    refreshed_gain_evaluations: int = 0
    update_gain_evaluations: int = 0
    heap_pops: int = 0
    heap_reinsertions: int = 0
    certifications: int = 0
    support_work: int = 0

    def performance(self, elapsed_seconds: float) -> EntropySelectionPerformance:
        return EntropySelectionPerformance(
            gain_evaluations=self.gain_evaluations,
            initial_gain_evaluations=self.initial_gain_evaluations,
            refreshed_gain_evaluations=self.refreshed_gain_evaluations,
            update_gain_evaluations=self.update_gain_evaluations,
            heap_pops=self.heap_pops,
            heap_reinsertions=self.heap_reinsertions,
            certifications=self.certifications,
            support_work=self.support_work,
            elapsed_seconds=elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class _PreparedSelection:
    state: EntropyObjectiveState
    contributions: SparseCandidateContributions
    anchor_ids: tuple[str, ...]
    structure_ids: tuple[str | None, ...]


@dataclass(slots=True)
class _LazyRecord:
    candidate_index: int
    exact_gain: float
    upper_bound: float
    epoch: int
    version: int = 0


def _resolve_optimizer_parameters(
    beta: float | int | None,
    budget: int | float | None,
    K: int | None,
) -> tuple[Any, Any]:
    """Accept the #133 ``beta, budget`` order and the documented ``K, beta`` order."""

    if K is not None:
        if budget is not None:
            raise ValueError("supply only one of budget or K")
        budget = K
    if beta is None or budget is None:
        raise ValueError("entropy greedy selection requires beta and final K/budget")
    if (
        isinstance(beta, (int, np.integer))
        and not isinstance(beta, bool)
        and not isinstance(budget, (int, np.integer))
    ):
        beta, budget = budget, beta
    return beta, budget


def _canonical_anchor_ids(anchors: Iterable[str] | str | None) -> tuple[str, ...]:
    if anchors is None:
        return ()
    if isinstance(anchors, str):
        values = (anchors,)
    else:
        try:
            values = tuple(anchors)
        except TypeError as exc:
            raise ValueError("anchors must be an iterable of candidate IDs") from exc
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("anchors must contain non-blank candidate IDs")
    if len(set(values)) != len(values):
        raise ValueError("anchors must contain unique candidate IDs")
    return tuple(sorted(values))


def _structure_id_values(
    candidate_ids: tuple[str, ...],
    candidate_structure_ids: Mapping[str, str] | None,
    structure_ids: Mapping[str, str] | None,
) -> tuple[str | None, ...]:
    if candidate_structure_ids is not None and structure_ids is not None:
        raise ValueError("supply only one structure provenance mapping")
    mapping = candidate_structure_ids if candidate_structure_ids is not None else structure_ids
    if mapping is None:
        return (None,) * len(candidate_ids)
    if not isinstance(mapping, Mapping):
        raise TypeError("structure provenance must be a candidate_id to structure_id mapping")
    if tuple(mapping) != candidate_ids:
        raise ValueError("structure provenance must be ordered exactly like candidate_ids")
    values = tuple(mapping[candidate_id] for candidate_id in candidate_ids)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("structure provenance values must be non-blank strings")
    return values


def _prepare_selection(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta: float | int | None,
    budget: int | float | None,
    K: int | None,
    anchors: Iterable[str] | str | None,
    candidate_structure_ids: Mapping[str, str] | None,
    structure_ids: Mapping[str, str] | None,
) -> _PreparedSelection:
    beta_value, budget_value = _resolve_optimizer_parameters(beta, budget, K)
    anchor_ids = _canonical_anchor_ids(anchors)
    # This is deliberately the only objective initialization boundary.  It
    # validates the pool, graph, q_C rows, beta, K and anchor membership.
    state = initialize_entropy_objective(
        pool,
        contributions,
        beta_value,
        anchors=anchor_ids,
        budget=budget_value,
    )
    structure_values = _structure_id_values(
        tuple(pool.candidate_ids), candidate_structure_ids, structure_ids
    )
    return _PreparedSelection(state, contributions, anchor_ids, structure_values)


def _checked_gain(
    state: EntropyObjectiveState,
    contributions: SparseCandidateContributions,
    candidate_index: int,
    counters: _SelectionCounters,
    phase: str,
) -> float:
    row = contributions.candidate_row(candidate_index)
    gain = float(marginal_gain(state, contributions, candidate_index))
    counters.gain_evaluations += 1
    counters.support_work += int(row.target_indices.size)
    if phase == "initial":
        counters.initial_gain_evaluations += 1
    elif phase == "refresh":
        counters.refreshed_gain_evaluations += 1
    elif phase == "update":
        counters.update_gain_evaluations += 1
    else:
        raise RuntimeError(f"unknown gain evaluation phase: {phase!r}")
    if not math.isfinite(gain) or gain <= 0.0:
        raise EntropySelectionError(
            f"candidate {state.candidate_ids[candidate_index]!r} returned an invalid gain"
        )
    return gain


def _apply_selected_candidate(
    state: EntropyObjectiveState,
    contributions: SparseCandidateContributions,
    candidate_index: int,
    selected_gain: float,
    counters: _SelectionCounters,
) -> None:
    previous_objective = state.objective
    row = contributions.candidate_row(candidate_index)
    apply_candidate(state, contributions, candidate_index)
    counters.gain_evaluations += 1
    counters.update_gain_evaluations += 1
    counters.support_work += int(row.target_indices.size)
    expected = previous_objective + selected_gain
    if not math.isclose(
        state.objective,
        expected,
        rel_tol=0.0,
        abs_tol=state.numerical_tolerance,
    ):
        raise EntropySelectionError("objective update disagrees with the certified marginal gain")


def _shannon_entropy(state: EntropyObjectiveState) -> float:
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        try:
            value = -float(np.sum(state.probabilities * np.log(state.probabilities)))
        except FloatingPointError as exc:
            raise EntropySelectionError("Shannon entropy became non-finite") from exc
    if not math.isfinite(value):
        raise EntropySelectionError("Shannon entropy became non-finite")
    return value


def _prefix_diagnostics(
    state: EntropyObjectiveState,
    shannon_entropy: float | None,
) -> tuple[float | None, float | None]:
    if shannon_entropy is None:
        return None, None
    denominator = state.beta + state.selected_count
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        try:
            cross_entropy = float(np.log(denominator) - state.objective)
        except FloatingPointError as exc:
            raise EntropySelectionError("prefix entropy diagnostic became non-finite") from exc
    forward_kl = cross_entropy - shannon_entropy
    if not math.isfinite(cross_entropy) or not math.isfinite(forward_kl):
        raise EntropySelectionError("prefix entropy diagnostic became non-finite")
    if forward_kl < -state.numerical_tolerance:
        raise EntropySelectionError("prefix forward KL diagnostic became materially negative")
    return cross_entropy, max(0.0, forward_kl)


def _anchor_steps(
    prepared: _PreparedSelection,
    shannon_entropy: float | None,
) -> list[EntropySelectionStep]:
    state = prepared.state
    cross_entropy, forward_kl = _prefix_diagnostics(state, shannon_entropy)
    return [
        EntropySelectionStep(
            candidate_id=candidate_id,
            structure_id=prepared.structure_ids[state.candidate_ids.index(candidate_id)],
            acquisition_iteration=0,
            reason="anchor",
            marginal_gain=None,
            objective=state.objective,
            anchor_normalized_objective=0.0,
            selected_count=state.selected_count,
            cross_entropy=cross_entropy,
            forward_kl=forward_kl,
            state_fingerprint=state.fingerprint,
        )
        for candidate_id in prepared.anchor_ids
    ]


def _greedy_step(
    prepared: _PreparedSelection,
    candidate_index: int,
    gain: float,
    iteration: int,
    shannon_entropy: float | None,
) -> EntropySelectionStep:
    state = prepared.state
    cross_entropy, forward_kl = _prefix_diagnostics(state, shannon_entropy)
    return EntropySelectionStep(
        candidate_id=state.candidate_ids[candidate_index],
        structure_id=prepared.structure_ids[candidate_index],
        acquisition_iteration=iteration,
        reason="greedy",
        marginal_gain=gain,
        objective=state.objective,
        anchor_normalized_objective=state.anchor_normalized_objective,
        selected_count=state.selected_count,
        cross_entropy=cross_entropy,
        forward_kl=forward_kl,
        state_fingerprint=state.fingerprint,
    )


def _better_pair(gain: float, candidate_id: str, other_gain: float, other_id: str) -> bool:
    """Shared exact ordering: larger float first, then lexicographically smaller ID."""

    return gain > other_gain or (gain == other_gain and candidate_id < other_id)


def _upper_bound(gain: float) -> float:
    """Return the exact cached float used by the shared gain ordering policy.

    No near-tie tolerance or arbitrary padding is used.  If a later exact
    evaluation exceeds this cached value, certification fails explicitly.
    """

    return gain


def _progress_log(
    logger: logging.Logger,
    label: str,
    completed: int,
    total: int,
    started: float,
    gain_evaluations: int,
    previous_marker: int = 0,
) -> int:
    if total < 1:
        return 0
    if total < 100:
        marker = completed
        should_log = completed > 0
    else:
        marker = min(100, (completed * 100) // total)
        should_log = marker > 0
    if not should_log or marker <= previous_marker:
        return previous_marker
    elapsed = max(0.0, time.perf_counter() - started)
    throughput = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = total - completed
    eta = remaining / throughput if throughput > 0.0 else math.inf
    logger.info(
        "%s progress: %d/%d (%d%%), elapsed=%.3fs, throughput=%.3f/s, ETA=%s, gain_evaluations=%d",
        label,
        completed,
        total,
        marker if total >= 100 else int(completed * 100 / total),
        elapsed,
        throughput,
        f"{eta:.3f}s" if math.isfinite(eta) else "unknown",
        gain_evaluations,
    )
    return marker


def _result(
    method: str,
    prepared: _PreparedSelection,
    acquisition_order: list[str],
    steps: list[EntropySelectionStep],
    counters: _SelectionCounters,
    started: float,
) -> EntropySelectionResult:
    state = prepared.state
    if state.selected_count != state.budget:
        raise EntropySelectionError("greedy optimizer stopped before its fixed budget")
    diagnostics = objective_diagnostics(state)
    normalized_support = tuple(float(value) for value in state.s / (state.beta + state.budget))
    selected_indices = tuple(sorted(state.selected_indices))
    selected_ids = tuple(state.candidate_ids[index] for index in selected_indices)
    provenance_values: list[tuple[str, str]] = []
    for candidate_index in selected_indices:
        structure_id = prepared.structure_ids[candidate_index]
        if structure_id is not None:
            provenance_values.append((state.candidate_ids[candidate_index], structure_id))
    provenance = tuple(provenance_values)
    history = EntropySelectionHistory(tuple(acquisition_order), tuple(steps))
    return EntropySelectionResult(
        method=method,
        method_version=GREEDY_SELECTION_VERSION,
        budget=state.budget,
        beta=state.beta,
        anchor_ids=prepared.anchor_ids,
        acquisition_order=tuple(acquisition_order),
        selected_candidate_ids=selected_ids,
        selected_indices=selected_indices,
        history=history,
        anchor_objective=state.anchor_objective,
        final_objective=state.objective,
        final_anchor_normalized_objective=state.anchor_normalized_objective,
        final_cross_entropy=diagnostics.cross_entropy,
        final_shannon_entropy=diagnostics.shannon_entropy,
        final_forward_kl=diagnostics.kl_divergence,
        pool_fingerprint=state.pool_fingerprint,
        graph_fingerprint=state.graph_fingerprint,
        kernel_operator_fingerprint=state.kernel_operator_fingerprint,
        contributions_fingerprint=state.contributions_fingerprint,
        state_fingerprint=state.fingerprint,
        performance=counters.performance(time.perf_counter() - started),
        final_normalized_support=normalized_support,
        structure_provenance=provenance,
    )


def _start_log(method: str, prepared: _PreparedSelection, logger: logging.Logger) -> None:
    state = prepared.state
    logger.info(
        "%s start: M=%d, N=%d, K=%d, anchors=%d, remaining=%d, beta=%.17g, "
        "kernel_operator=%s, contributions=%s, tie_rule=largest_exact_gain_then_lexicographic_id",
        method,
        len(state.candidate_ids),
        state.n_targets,
        state.budget,
        len(prepared.anchor_ids),
        state.budget - state.selected_count,
        state.beta,
        state.kernel_operator_fingerprint,
        state.contributions_fingerprint,
    )


def _run_full_greedy(
    prepared: _PreparedSelection,
    include_diagnostics: bool,
) -> EntropySelectionResult:
    started = time.perf_counter()
    state = prepared.state
    counters = _SelectionCounters()
    logger = LOGGER
    _start_log("full_greedy", prepared, logger)
    shannon = _shannon_entropy(state) if include_diagnostics else None
    steps = _anchor_steps(prepared, shannon)
    acquisition_order = list(prepared.anchor_ids)
    remaining = set(range(len(state.candidate_ids))) - set(state.selected_indices)
    required = state.budget - state.selected_count
    progress_marker = 0
    if required == 0:
        logger.info("full_greedy no-op: anchors already cover the requested budget")
        return _result("full_greedy", prepared, acquisition_order, steps, counters, started)

    for round_index in range(required):
        best_index: int | None = None
        best_gain = -math.inf
        for candidate_index in sorted(remaining, key=state.candidate_ids.__getitem__):
            gain = _checked_gain(
                state,
                prepared.contributions,
                candidate_index,
                counters,
                "initial" if round_index == 0 else "refresh",
            )
            candidate_id = state.candidate_ids[candidate_index]
            if best_index is None or _better_pair(
                gain,
                candidate_id,
                best_gain,
                state.candidate_ids[best_index],
            ):
                best_index = candidate_index
                best_gain = gain
        if best_index is None:
            raise EntropySelectionError("full greedy could not find an unselected candidate")
        _apply_selected_candidate(
            state,
            prepared.contributions,
            best_index,
            best_gain,
            counters,
        )
        remaining.remove(best_index)
        acquisition_order.append(state.candidate_ids[best_index])
        steps.append(
            _greedy_step(
                prepared,
                best_index,
                best_gain,
                round_index + 1,
                shannon,
            )
        )
        progress_marker = _progress_log(
            logger,
            "full_greedy acquisition",
            round_index + 1,
            required,
            started,
            counters.gain_evaluations,
            progress_marker,
        )
    result = _result("full_greedy", prepared, acquisition_order, steps, counters, started)
    logger.info(
        "full_greedy complete: runtime=%.3fs, gain_evaluations=%d, refreshed=%d, "
        "final_F=%.17g, cross_entropy=%.17g, forward_KL=%.17g%s",
        result.performance.elapsed_seconds,
        result.performance.gain_evaluations,
        result.performance.refreshed_gain_evaluations,
        result.final_objective,
        result.final_cross_entropy,
        result.final_forward_kl,
        ", K=M selected all candidates" if state.budget == len(state.candidate_ids) else "",
    )
    return result


def _run_lazy_greedy(
    prepared: _PreparedSelection,
    include_diagnostics: bool,
) -> EntropySelectionResult:
    started = time.perf_counter()
    state = prepared.state
    counters = _SelectionCounters()
    logger = LOGGER
    _start_log("lazy_greedy", prepared, logger)
    shannon = _shannon_entropy(state) if include_diagnostics else None
    steps = _anchor_steps(prepared, shannon)
    acquisition_order = list(prepared.anchor_ids)
    remaining = set(range(len(state.candidate_ids))) - set(state.selected_indices)
    required = state.budget - state.selected_count
    if required == 0:
        logger.info("lazy_greedy no-op: anchors already cover the requested budget")
        return _result("lazy_greedy", prepared, acquisition_order, steps, counters, started)

    heap: list[tuple[float, str, int, int]] = []
    records: dict[int, _LazyRecord] = {}
    initial_total = len(remaining)
    progress_marker = 0
    for candidate_index in sorted(remaining, key=state.candidate_ids.__getitem__):
        gain = _checked_gain(
            state,
            prepared.contributions,
            candidate_index,
            counters,
            "initial",
        )
        record = _LazyRecord(candidate_index, gain, _upper_bound(gain), 0)
        records[candidate_index] = record
        heapq.heappush(
            heap,
            (-record.upper_bound, state.candidate_ids[candidate_index], candidate_index, 0),
        )
        progress_marker = _progress_log(
            logger,
            "lazy_greedy initial gain scan",
            counters.initial_gain_evaluations,
            initial_total,
            started,
            counters.gain_evaluations,
            progress_marker,
        )

    epoch = 0
    acquisition_progress_marker = 0
    for acquisition_index in range(required):
        while True:
            if not heap:
                raise EntropySelectionCertificationError(
                    "lazy greedy queue became empty before the fixed budget was met"
                )
            _, candidate_id, candidate_index, version = heapq.heappop(heap)
            counters.heap_pops += 1
            record = records.get(candidate_index)
            if record is None or candidate_index not in remaining:
                raise EntropySelectionCertificationError(
                    "lazy greedy queue contains an unknown candidate"
                )
            if record.version != version or candidate_id != state.candidate_ids[candidate_index]:
                raise EntropySelectionCertificationError(
                    "lazy greedy queue contains a stale record"
                )

            if record.epoch < epoch:
                previous_bound = record.upper_bound
                gain = _checked_gain(
                    state,
                    prepared.contributions,
                    candidate_index,
                    counters,
                    "refresh",
                )
                if gain > previous_bound:
                    raise EntropySelectionCertificationError(
                        "fresh marginal gain exceeded its cached upper bound; "
                        "winner cannot be certified safely"
                    )
                record.exact_gain = gain
                record.upper_bound = _upper_bound(gain)
                record.epoch = epoch
                record.version += 1
            else:
                gain = record.exact_gain

            if heap:
                _, competitor_id, competitor_index, competitor_version = heap[0]
                competitor = records.get(competitor_index)
                if (
                    competitor is None
                    or competitor.version != competitor_version
                    or competitor_index not in remaining
                    or competitor_id != state.candidate_ids[competitor_index]
                ):
                    raise EntropySelectionCertificationError(
                        "lazy greedy queue contains a corrupt competitor record"
                    )
                certified = _better_pair(
                    gain,
                    candidate_id,
                    competitor.upper_bound,
                    competitor_id,
                ) or (gain == competitor.upper_bound and candidate_id == competitor_id)
            else:
                certified = True
            if certified:
                counters.certifications += 1
                break
            record.version += 1
            heapq.heappush(
                heap,
                (-record.upper_bound, candidate_id, candidate_index, record.version),
            )
            counters.heap_reinsertions += 1
            logger.debug(
                "lazy_greedy certification deferred: candidate=%s gain=%.17g",
                candidate_id,
                gain,
            )

        _apply_selected_candidate(
            state,
            prepared.contributions,
            candidate_index,
            gain,
            counters,
        )
        remaining.remove(candidate_index)
        records.pop(candidate_index, None)
        epoch += 1
        acquisition_order.append(candidate_id)
        steps.append(
            _greedy_step(
                prepared,
                candidate_index,
                gain,
                acquisition_index + 1,
                shannon,
            )
        )
        acquisition_progress_marker = _progress_log(
            logger,
            "lazy_greedy acquisition",
            acquisition_index + 1,
            required,
            started,
            counters.gain_evaluations,
            acquisition_progress_marker,
        )
    result = _result("lazy_greedy", prepared, acquisition_order, steps, counters, started)
    logger.info(
        "lazy_greedy complete: runtime=%.3fs, gain_evaluations=%d, refreshed=%d, "
        "certifications=%d, final_F=%.17g, cross_entropy=%.17g, forward_KL=%.17g%s",
        result.performance.elapsed_seconds,
        result.performance.gain_evaluations,
        result.performance.refreshed_gain_evaluations,
        result.performance.certifications,
        result.final_objective,
        result.final_cross_entropy,
        result.final_forward_kl,
        ", K=M selected all candidates" if state.budget == len(state.candidate_ids) else "",
    )
    return result


def _run_optimizer(
    method: str,
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta: float | int | None,
    budget: int | float | None,
    K: int | None,
    anchors: Iterable[str] | str | None,
    candidate_structure_ids: Mapping[str, str] | None,
    structure_ids: Mapping[str, str] | None,
    include_diagnostics: bool,
) -> EntropySelectionResult:
    prepared = _prepare_selection(
        pool,
        contributions,
        beta,
        budget,
        K,
        anchors,
        candidate_structure_ids,
        structure_ids,
    )
    if method == "full_greedy":
        return _run_full_greedy(prepared, include_diagnostics)
    if method == "lazy_greedy":
        return _run_lazy_greedy(prepared, include_diagnostics)
    raise ValueError(f"unknown entropy greedy method: {method!r}")


def full_greedy(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta: float | int | None = None,
    budget: int | float | None = None,
    anchors: Iterable[str] | str | None = (),
    *,
    K: int | None = None,
    anchor_ids: Iterable[str] | str | None = None,
    candidate_structure_ids: Mapping[str, str] | None = None,
    structure_ids: Mapping[str, str] | None = None,
    include_diagnostics: bool = True,
) -> EntropySelectionResult:
    """Run the deterministic full-greedy sparse objective reference."""

    if anchor_ids is not None:
        if anchors not in ((), None):
            raise ValueError("supply only one of anchors or anchor_ids")
        anchors = anchor_ids
    try:
        return _run_optimizer(
            "full_greedy",
            pool,
            contributions,
            beta,
            budget,
            K,
            anchors,
            candidate_structure_ids,
            structure_ids,
            include_diagnostics,
        )
    except Exception as exc:
        LOGGER.error("full_greedy failed: %s", exc)
        raise


def lazy_greedy(
    pool: EntropyPool,
    contributions: SparseCandidateContributions,
    beta: float | int | None = None,
    budget: int | float | None = None,
    anchors: Iterable[str] | str | None = (),
    *,
    K: int | None = None,
    anchor_ids: Iterable[str] | str | None = None,
    candidate_structure_ids: Mapping[str, str] | None = None,
    structure_ids: Mapping[str, str] | None = None,
    include_diagnostics: bool = True,
) -> EntropySelectionResult:
    """Run production lazy greedy with exact tie-aware certification."""

    if anchor_ids is not None:
        if anchors not in ((), None):
            raise ValueError("supply only one of anchors or anchor_ids")
        anchors = anchor_ids
    try:
        return _run_optimizer(
            "lazy_greedy",
            pool,
            contributions,
            beta,
            budget,
            K,
            anchors,
            candidate_structure_ids,
            structure_ids,
            include_diagnostics,
        )
    except Exception as exc:
        LOGGER.error("lazy_greedy failed: %s", exc)
        raise


run_full_greedy = full_greedy
run_lazy_greedy = lazy_greedy
select_full_greedy = full_greedy
select_lazy_greedy = lazy_greedy


class InformationEntropySelectionAlgorithm:
    """Registry adapter for the single production sparse entropy pipeline.

    Representation calculation remains owned by ``SelectionStage``.  This
    adapter owns the remaining finite-pool scientific sequence and delegates
    greedy acquisition to the same public ``lazy_greedy``/``full_greedy``
    implementations used by the reference tests.
    """

    algorithm_id = "information_entropy"
    algorithm_version = "information-entropy-wendland-v1"

    @staticmethod
    def _bandwidth_settings(options: Mapping[str, Any]) -> EntropyBandwidthSettings:
        supplied = options.get("entropy_bandwidth")
        if isinstance(supplied, EntropyBandwidthSettings):
            return supplied
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("entropy_bandwidth must be EntropyBandwidthSettings or a mapping")
            return EntropyBandwidthSettings(**dict(supplied))
        entropy = options.get("entropy_config")
        bandwidth = getattr(entropy, "bandwidth", None)
        if isinstance(bandwidth, EntropyBandwidthSettings):
            return bandwidth
        if bandwidth is not None:
            return EntropyBandwidthSettings(
                mode=str(bandwidth.mode),
                k=bandwidth.k,
                c=bandwidth.c,
                k_candidates=tuple(bandwidth.k_candidates),
                c_candidates=tuple(bandwidth.c_candidates),
                backend=str(bandwidth.backend),
                metric=str(bandwidth.metric),
                chunk_size=int(bandwidth.chunk_size),
                max_neighbour_entries=(
                    None
                    if bandwidth.max_neighbour_entries is None
                    else int(bandwidth.max_neighbour_entries)
                ),
                max_index_bytes=(
                    None if bandwidth.max_index_bytes is None else int(bandwidth.max_index_bytes)
                ),
                max_radius_query_bytes=(
                    None
                    if bandwidth.max_radius_query_bytes is None
                    else int(bandwidth.max_radius_query_bytes)
                ),
                max_calibration_work_bytes=(
                    None
                    if bandwidth.max_calibration_work_bytes is None
                    else int(bandwidth.max_calibration_work_bytes)
                ),
                calibration_batch_size=(
                    None
                    if bandwidth.calibration_batch_size is None
                    else int(bandwidth.calibration_batch_size)
                ),
            )
        return EntropyBandwidthSettings()

    @staticmethod
    def _entropy_value(options: Mapping[str, Any], name: str, default: Any) -> Any:
        entropy = options.get("entropy_config")
        value = getattr(entropy, name, default)
        return options.get(name, value)

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        if request.algorithm_id != self.algorithm_id:
            raise ValueError(
                f"Request algorithm {request.algorithm_id!r} does not match information entropy"
            )
        candidate_count = len(request.candidate_ids)
        if request.target_count > candidate_count:
            raise ValueError(
                "information-entropy target_train_count exceeds available candidates: "
                f"requested K={request.target_count}, available M={candidate_count}"
            )
        anchors = tuple(request.anchor_ids)
        if len(set(anchors)) > request.target_count:
            raise ValueError("anchor count exceeds information-entropy target count")
        local = request.options.get("local_representation")
        if not isinstance(local, LocalEnvironmentRepresentation):
            raise ValueError(
                "information-entropy selection requires a local environment representation"
            )
        pool = build_entropy_pool(local)
        if tuple(pool.candidate_ids) != request.candidate_ids:
            raise ValueError("local representation candidate order does not match selection input")
        bandwidth_settings = self._bandwidth_settings(request.options)
        resource_budget = request.options.get("resource_budget")
        if resource_budget is not None and not isinstance(resource_budget, ResourceBudgetService):
            raise TypeError("resource_budget must be a ResourceBudgetService")
        runtime_budget = resource_budget or build_resource_budget()
        calibration = calibrate_bandwidth(
            pool,
            bandwidth_settings,
            resource_budget=runtime_budget,
        )
        remaining_budget = runtime_budget.remaining_managed_budget
        if remaining_budget is None:
            raise ValueError(
                runtime_budget.unknown_memory_message("information-entropy selection")
            )
        radius_query_bytes = min(
            bandwidth_settings.max_radius_query_bytes or max(1, remaining_budget // 2),
            max(1, remaining_budget),
        )
        logger = LOGGER
        logger.info(
            "Entropy bandwidth calibrated: k=%d, c=%.17g, LOO objective=%s",
            calibration.selected.k,
            calibration.selected.c,
            calibration.objective if calibration.objective is not None else "n/a",
        )
        bandwidth = calibration.selected
        # The production operator streams exact normalized source columns
        # directly into candidate PMFs.  max_edges/max_graph_* remain limits
        # for the bounded graph reference API and do not constrain implicit E.
        contributions, execution = build_streamed_candidate_contributions(
            pool,
            bandwidth,
            chunk_size=bandwidth_settings.chunk_size,
            max_entries=self._entropy_value(request.options, "max_entries", None),
            max_contribution_bytes=self._entropy_value(
                request.options, "max_contribution_bytes", None
            ),
            max_spool_bytes=self._entropy_value(
                request.options, "max_contribution_spool_bytes", None
            ),
            max_radius_query_bytes=radius_query_bytes,
            resource_budget=runtime_budget,
        )
        method = str(self._entropy_value(request.options, "optimizer_method", "lazy_greedy"))
        optimizer = full_greedy if method == "full_greedy" else lazy_greedy
        if method not in {"full_greedy", "lazy_greedy"}:
            raise ValueError(f"unsupported information-entropy optimizer method: {method!r}")
        supplied_provenance = request.options.get("candidate_structure_ids")
        structure_provenance = (
            {
                candidate_id: str(supplied_provenance[candidate_id])
                for candidate_id in request.candidate_ids
            }
            if isinstance(supplied_provenance, Mapping)
            else None
        )
        entropy_result = optimizer(
            pool,
            contributions,
            beta=float(self._entropy_value(request.options, "beta", 1.0)),
            K=request.target_count,
            anchors=anchors,
            candidate_structure_ids=structure_provenance,
        )
        diagnostics = {
            "entropy_result": entropy_result,
            "calibration": calibration,
            "graph": execution,
            "execution": execution,
            "contributions": contributions,
            "pool": pool,
        }
        return SelectionAlgorithmResult(
            algorithm_id=self.algorithm_id,
            algorithm_version=self.algorithm_version,
            selected_indices=entropy_result.selected_indices,
            selected_candidate_ids=entropy_result.selected_candidate_ids,
            diagnostics=(diagnostics,),
            minimum_distance=None,
            algorithm_result=entropy_result,
        )


__all__ = [
    "EntropySelectionCertificationError",
    "EntropySelectionError",
    "InformationEntropySelectionAlgorithm",
    "SelectionCertificationError",
    "full_greedy",
    "lazy_greedy",
    "run_full_greedy",
    "run_lazy_greedy",
    "select_full_greedy",
    "select_lazy_greedy",
]
