"""Focused full- and lazy-greedy information-entropy regressions."""

from dataclasses import replace

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.models import (
    SparseCandidateContributions,
)
from nepflow.stages.selection.algorithms.information_entropy.selector import (
    full_greedy,
    lazy_greedy,
)


def _fixture(*, coincident: bool = False):
    pool = build_entropy_pool(
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64),
        row_candidate_ids=["z", "a", "m"],
        candidate_ids=["z", "a", "m"],
    )
    if coincident:
        target_indices = np.asarray([0, 0, 0], dtype=np.int64)
    else:
        target_indices = np.asarray([0, 1, 2], dtype=np.int64)
    contributions = SparseCandidateContributions(
        candidate_indptr=np.asarray([0, 1, 2, 3], dtype=np.int64),
        target_indices=target_indices,
        values=np.ones(3, dtype=np.float64),
        candidate_ids=pool.candidate_ids,
        graph_fingerprint="selector-test-graph",
        fingerprint="selector-test-contributions",
        row_count=pool.probabilities.size,
        pool_fingerprint=pool.fingerprint,
    )
    return pool, contributions


def test_full_and_lazy_greedy_match_exact_sequence_and_history() -> None:
    pool, contributions = _fixture()
    full = full_greedy(pool, contributions, beta=0.25, K=2)
    lazy = lazy_greedy(pool, contributions, beta=0.25, K=2)

    assert full.acquisition_order == lazy.acquisition_order
    assert full.selected_candidate_ids == lazy.selected_candidate_ids
    assert full.selected_indices == lazy.selected_indices
    assert [step.marginal_gain for step in full.history.steps] == pytest.approx(
        [step.marginal_gain for step in lazy.history.steps]
    )
    assert [step.objective for step in full.history.steps] == pytest.approx(
        [step.objective for step in lazy.history.steps]
    )
    assert full.final_objective == pytest.approx(lazy.final_objective)
    assert full.final_cross_entropy == pytest.approx(lazy.final_cross_entropy)
    assert full.final_forward_kl == pytest.approx(lazy.final_forward_kl)


def test_equal_gains_use_candidate_id_not_pool_row_order() -> None:
    pool, contributions = _fixture(coincident=True)
    result = lazy_greedy(pool, contributions, beta=0.25, budget=2)

    assert result.acquisition_order == ("a", "m")
    assert result.history.steps[0].marginal_gain is not None
    assert result.history.steps[1].marginal_gain is not None


def test_anchors_are_canonical_and_consume_budget_without_gain_claims() -> None:
    pool, contributions = _fixture()
    result = full_greedy(pool, contributions, beta=0.25, budget=2, anchors=("z",))

    assert result.acquisition_order[0] == "z"
    assert result.anchor_count == 1
    assert result.history.steps[0].reason == "anchor"
    assert result.history.steps[0].marginal_gain is None
    assert result.history.steps[1].reason == "greedy"


@pytest.mark.parametrize("budget", [-1, 4, 1.5, True])
def test_invalid_fixed_budget_is_rejected(budget: object) -> None:
    pool, contributions = _fixture()
    with pytest.raises(ValueError, match="budget|integer"):
        full_greedy(pool, contributions, beta=0.25, budget=budget)  # type: ignore[arg-type]


def test_stale_pool_identity_is_rejected_before_greedy_work() -> None:
    pool, contributions = _fixture()
    stale = replace(contributions, pool_fingerprint="different-pool")
    with pytest.raises(ValueError, match="different entropy pool"):
        lazy_greedy(pool, stale, beta=0.25, budget=2)


def test_zero_budget_has_no_gain_evaluations() -> None:
    pool, contributions = _fixture()
    result = lazy_greedy(pool, contributions, beta=0.25, budget=0)

    assert result.acquisition_order == ()
    assert result.performance.gain_evaluations == 0
