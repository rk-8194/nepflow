"""Independent regression tests for the exact sparse information objective."""

from dataclasses import replace

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    aggregate_candidate_contributions,
    build_sparse_atomic_kernel_graph,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyPool,
    FrozenBandwidths,
    SparseCandidateContributions,
)
from nepflow.stages.selection.algorithms.information_entropy.objective import (
    anchor_normalized_objective,
    apply_candidate,
    cross_entropy,
    forward_kl_divergence,
    initialize_entropy_objective,
    marginal_gain,
    objective_diagnostics,
    objective_value,
    recompute_objective,
    shannon_entropy,
)


def _fixture() -> tuple[EntropyPool, SparseCandidateContributions]:
    descriptors = np.asarray([[0.0], [0.0], [1.0], [2.0], [2.0], [4.0]], dtype=np.float64)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "a", "b", "c", "c", "c"],
        candidate_ids=["a", "b", "c"],
    )
    frozen = FrozenBandwidths(
        radii=np.ones(len(descriptors), dtype=np.float64),
        bandwidths=np.full(len(descriptors), 2.5, dtype=np.float64),
        k=1,
        c=1.0,
        neighbour_fingerprint="objective-test-neighbours",
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint="objective-test-bandwidths",
    )
    graph = build_sparse_atomic_kernel_graph(pool, frozen, chunk_size=1)
    return pool, aggregate_candidate_contributions(graph)


def _dense_contributions(contributions: SparseCandidateContributions) -> np.ndarray:
    dense = np.zeros((contributions.n_targets, contributions.n_candidates), dtype=np.float64)
    for candidate_index, row in enumerate(contributions.iter_candidates()):
        dense[row.target_indices, candidate_index] = row.values
    return dense


def test_pool_target_measure_and_real_sparse_q_are_consumed_as_is() -> None:
    pool, contributions = _fixture()
    expected = np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0])

    np.testing.assert_array_equal(pool.probabilities, expected)
    np.testing.assert_allclose(
        [np.sum(pool.probabilities[pool.row_candidate_indices == index]) for index in range(3)],
        [1.0 / 3.0] * 3,
    )
    dense = _dense_contributions(contributions)
    np.testing.assert_allclose(np.sum(dense, axis=0), 1.0)
    assert np.count_nonzero(dense[:, 0]) == contributions.candidate_row("a").target_indices.size
    assert np.array_equal(pool.descriptors[0], pool.descriptors[1])
    assert pool.row_candidate_indices[0] != pool.row_candidate_indices[2]


def test_empty_and_order_independent_anchor_initialisation() -> None:
    pool, contributions = _fixture()
    beta = 0.25
    empty = initialize_entropy_objective(pool, contributions, beta, anchors=(), budget=3)
    np.testing.assert_array_equal(empty.s, beta * pool.probabilities)
    assert empty.selected_ids == ()
    assert empty.selected_count == 0
    assert np.sum(empty.s) == pytest.approx(beta)

    first = initialize_entropy_objective(pool, contributions, beta, anchors=("c", "a"), budget=3)
    second = initialize_entropy_objective(pool, contributions, beta, anchors=("a", "c"), budget=3)
    q = _dense_contributions(contributions)
    expected = beta * pool.probabilities + q[:, 0] + q[:, 2]

    assert first.selected_ids == second.selected_ids == ("a", "c")
    np.testing.assert_array_equal(first.s, expected)
    np.testing.assert_array_equal(first.s, second.s)
    assert first.fingerprint == second.fingerprint
    assert np.sum(first.s) == pytest.approx(beta + 2.0)


def test_objective_diagnostics_use_current_prefix_normalisation() -> None:
    pool, contributions = _fixture()
    beta = 0.25
    state = initialize_entropy_objective(pool, contributions, beta, anchors=("a",), budget=3)
    diagnostics = objective_diagnostics(state)
    expected_cross_entropy = -float(
        np.sum(pool.probabilities * np.log(state.s / (beta + 1.0)), dtype=np.float64)
    )
    expected_shannon = -float(
        np.sum(pool.probabilities * np.log(pool.probabilities), dtype=np.float64)
    )

    assert diagnostics.selected_count == 1
    assert diagnostics.budget == 3
    assert diagnostics.is_final is False
    assert diagnostics.normalized_prefix_mass == pytest.approx(1.0)
    assert diagnostics.cross_entropy == pytest.approx(expected_cross_entropy)
    assert diagnostics.shannon_entropy == pytest.approx(expected_shannon)
    assert diagnostics.cross_entropy == pytest.approx(np.log(beta + 1.0) - state.objective)
    assert diagnostics.kl_divergence == pytest.approx(
        diagnostics.cross_entropy - diagnostics.shannon_entropy
    )
    assert cross_entropy(state) == pytest.approx(diagnostics.cross_entropy)
    assert shannon_entropy(state) == pytest.approx(diagnostics.shannon_entropy)
    assert forward_kl_divergence(state) == pytest.approx(diagnostics.kl_divergence)


def test_sparse_marginal_gain_matches_dense_reference_without_mutation() -> None:
    pool, contributions = _fixture()
    state = initialize_entropy_objective(pool, contributions, 0.25, anchors=("a",), budget=3)
    q = _dense_contributions(contributions)
    old_support = state.s.copy()
    old_objective = state.objective
    old_fingerprint = state.fingerprint
    old_probabilities = pool.probabilities.copy()
    old_values = contributions.values.copy()

    gain = marginal_gain(state, contributions, "b")
    support = np.flatnonzero(q[:, 1] > 0.0)
    dense_gain = float(
        np.sum(
            (pool.probabilities[support] * np.log1p(q[support, 1] / old_support[support])),
            dtype=np.float64,
        )
    )
    full_difference = float(
        np.sum(
            pool.probabilities * np.log(old_support + q[:, 1])
            - pool.probabilities * np.log(old_support),
            dtype=np.float64,
        )
    )

    assert gain == pytest.approx(dense_gain)
    assert gain == pytest.approx(full_difference)
    np.testing.assert_array_equal(state.s, old_support)
    assert state.objective == old_objective
    assert state.fingerprint == old_fingerprint
    np.testing.assert_array_equal(pool.probabilities, old_probabilities)
    np.testing.assert_array_equal(contributions.values, old_values)


def test_update_is_sparse_cached_and_anchor_normalised() -> None:
    pool, contributions = _fixture()
    state = initialize_entropy_objective(pool, contributions, 0.25, anchors=("a",), budget=3)
    before = state.s.copy()
    gain = marginal_gain(state, contributions, "b")
    row = contributions.candidate_row("b")

    returned = apply_candidate(state, contributions, "b")

    assert returned is state
    assert state.selected_ids == ("a", "b")
    assert state.selected_count == 2
    np.testing.assert_array_equal(
        state.s[row.target_indices], before[row.target_indices] + row.values
    )
    untouched = np.ones(state.n_targets, dtype=bool)
    untouched[row.target_indices] = False
    np.testing.assert_array_equal(state.s[untouched], before[untouched])
    assert state.objective == pytest.approx(objective_value(state))
    assert state.objective == pytest.approx(recompute_objective(state))
    assert state.objective == pytest.approx(state.anchor_objective + gain)
    assert anchor_normalized_objective(state) == pytest.approx(gain)
    assert np.sum(state.s) == pytest.approx(0.25 + 2.0)


def test_final_state_uses_beta_plus_k_and_fixed_k_ranking_equivalence() -> None:
    pool, contributions = _fixture()
    state = initialize_entropy_objective(pool, contributions, 0.25, anchors=("a",), budget=3)
    apply_candidate(state, contributions, "b")
    apply_candidate(state, contributions, "c")
    diagnostics = objective_diagnostics(state)

    assert state.is_final
    assert diagnostics.is_final
    assert diagnostics.denominator == pytest.approx(3.25)
    assert diagnostics.normalized_prefix_mass == pytest.approx(1.0)
    assert diagnostics.cross_entropy == pytest.approx(np.log(3.25) - state.objective)
    assert diagnostics.kl_divergence >= -1.0e-12

    q = _dense_contributions(contributions)
    set_scores: list[tuple[tuple[int, ...], float, float]] = []
    beta = 0.25
    for selected in ((0, 1, 2),):
        support = beta * pool.probabilities + np.sum(q[:, selected], axis=1)
        objective = float(np.sum(pool.probabilities * np.log(support), dtype=np.float64))
        cross = float(
            -np.sum(pool.probabilities * np.log(support / (beta + len(selected))), dtype=np.float64)
        )
        set_scores.append((selected, objective, cross))
    assert set_scores[0][2] == pytest.approx(np.log(beta + 3.0) - set_scores[0][1])


def test_monotonicity_and_diminishing_returns_on_identical_states() -> None:
    pool, contributions = _fixture()
    empty = initialize_entropy_objective(pool, contributions, 0.25, anchors=(), budget=3)
    with_a = initialize_entropy_objective(pool, contributions, 0.25, anchors=("a",), budget=3)

    gain_b_empty = marginal_gain(empty, contributions, "b")
    gain_c_empty = marginal_gain(empty, contributions, "c")
    gain_b_with_a = marginal_gain(with_a, contributions, "b")
    gain_c_with_a = marginal_gain(with_a, contributions, "c")

    assert gain_b_empty >= 0.0
    assert gain_c_empty >= 0.0
    assert gain_b_with_a <= gain_b_empty + 1.0e-12
    assert gain_c_with_a <= gain_c_empty + 1.0e-12
    assert anchor_normalized_objective(empty) == pytest.approx(0.0)


@pytest.mark.parametrize("beta", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_beta_is_rejected(beta: float) -> None:
    pool, contributions = _fixture()
    with pytest.raises(ValueError, match="beta"):
        initialize_entropy_objective(pool, contributions, beta, anchors=(), budget=2)


def test_invalid_anchors_and_budget_are_rejected_without_state_creation() -> None:
    pool, contributions = _fixture()
    for anchors, budget, message in [
        (("a", "a"), 2, "duplicate"),
        (("missing",), 2, "unknown"),
        (("a", "b"), 1, "anchor count"),
    ]:
        with pytest.raises((ValueError, KeyError), match=message):
            initialize_entropy_objective(pool, contributions, 0.25, anchors=anchors, budget=budget)
    with pytest.raises(ValueError, match="budget"):
        initialize_entropy_objective(pool, contributions, 0.25, anchors=(), budget=4)


def test_duplicate_selection_and_stale_contribution_identity_fail_without_partial_mutation() -> (
    None
):
    pool, contributions = _fixture()
    state = initialize_entropy_objective(pool, contributions, 0.25, anchors=("a",), budget=2)
    old_support = state.s.copy()
    old_ids = state.selected_ids
    with pytest.raises(ValueError, match="already selected"):
        apply_candidate(state, contributions, "a")
    np.testing.assert_array_equal(state.s, old_support)
    assert state.selected_ids == old_ids

    stale = replace(contributions, pool_fingerprint="stale-pool")
    with pytest.raises(ValueError, match="different entropy pool"):
        initialize_entropy_objective(pool, stale, 0.25, anchors=(), budget=2)
